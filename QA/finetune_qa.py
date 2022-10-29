import comet_ml
import transformers
from datasets import load_dataset
import argparse
from qa_dataset import Dataset
import evaluate
import re

import torch

from sklearn.model_selection import train_test_split
from utils_qa import normalize_answer, normalize_answers, f1_score, exact_match_score, metric_max_over_ground_truths # from gsarti/it5


'''
----------------------------------------------------------------------------------------------------

                                        Parsing Arguments

----------------------------------------------------------------------------------------------------
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--logging_steps", type=int, default=100, required=False)
    parser.add_argument("--max_input_length", type=int, default=1024, required=False)
    parser.add_argument("--max_output_length", type=int, default=128, required=False)
    parser.add_argument("--learning_rate", type=float, default=1e-4, required=False)
    parser.add_argument("--dataloader_num_workers", type=int, default=8, required=False)
    parser.add_argument("--save_total_limit", type=int, default=5, required=False)
    parser.add_argument("--use_cuda", default=False, action="store_true", required=False)
    parser.add_argument("--fp16", default=False, action="store_true", required=False)
    parser.add_argument("--hub_model_id", type=str, default="", required=False)
    parser.add_argument("--push_to_hub", default=False, action="store_true", required=False)
    parser.add_argument("--context_token", type=str, default="[CONTEXT]", required=False)
    parser.add_argument("--question", type=str, default="[QUESTION]", required=False)

    parser.add_argument("--log_dir", type=str, default="", required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="", required=True)
    parser.add_argument("--dataset_name", type=str, default="", required=True)
    parser.add_argument("--model_path", type=str, default="", required=True)
    parser.add_argument("--tokenizer_path", type=str, default="", required=True)

    args = parser.parse_args()
    return args

args = parse_args()


'''
----------------------------------------------------------------------------------------------------

                                        Loading Data and Model

----------------------------------------------------------------------------------------------------
'''

model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

# extend tokenizer to add special tokens for QA
special_tokens_dict = {'additional_special_tokens': [args.context_token, args.question]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')
model.resize_token_embeddings(len(tokenizer))

# save tokenizer
tokenizer.save_pretrained(args.checkpoint_dir + "/tokenizer")

dataset = load_dataset(args.dataset_name)
data = dataset["train"]

print ("Len dataset: ", len(data))

# split data into train and validation
train_dataset, val_dataset = train_test_split(data, test_size=0.1)

train_questions = train_dataset["question"]
train_context = train_dataset["context"]
train_answers = train_dataset["answers"]
train_answers = [e["text"] for e in train_answers]
train_single_answers = [ e[0] for e in train_answers]

val_questions = val_dataset["question"]
val_context = val_dataset["context"]
val_answers = val_dataset["answers"]
val_answers = [e["text"] for e in val_answers]
val_single_answers = [ e[0] for e in val_answers ]

print ("Len train: ", len(train_questions))
print ("Len val: ", len(val_questions))


train_dataset = Dataset(
    contexts=train_context,
    questions=train_questions,
    answers=train_single_answers,
    tokenizer=tokenizer,
    context_token=args.context_token,
    question_token=args.question,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
)

val_dataset = Dataset(
    contexts=val_context,
    questions=val_questions,
    answers=val_single_answers,
    tokenizer=tokenizer,
    context_token=args.context_token,
    question_token=args.question,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
)


'''
----------------------------------------------------------------------------------------------------

                                        Training arguments

----------------------------------------------------------------------------------------------------
'''

# eval batch size is // 8 or 1, whichever is larger
#eval_batch_size = max(args.batch_size // 8, 1)
eval_batch_size = args.batch_size

training_arguments = transformers.TrainingArguments(
    output_dir=args.checkpoint_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=eval_batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=args.log_dir,
    logging_steps=args.logging_steps,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=args.learning_rate,
    dataloader_num_workers=args.dataloader_num_workers,
    save_total_limit=args.save_total_limit,
    no_cuda=not (args.use_cuda),
    fp16=args.fp16,
    metric_for_best_model="EM",
    greater_is_better=True,
    hub_model_id=args.hub_model_id,
    push_to_hub=args.push_to_hub,
)

'''
----------------------------------------------------------------------------------------------------

                                        Defining Metrics

----------------------------------------------------------------------------------------------------
'''

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
exact_match_metric = evaluate.load("exact_match")

def compute_metrics(pred):

    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = val_single_answers
    #label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    print (f"Comparing {len(pred_str)} predictions with {len(label_str)} labels")
    '''
    for i in range(len(pred_str)):
        print (f"Prediction: {pred_str[i]}")
        print (f"Label: {label_str[i]}")
        print ("-----------------------\n\n")
    '''
    print (f"Prediction: {pred_str[0]}")
    print (f"Label: {label_str[0]}")
    print ("-----------------------\n\n")

    rouge_output = rouge.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=False,
    )

    bertscore_output = bertscore.compute(predictions=pred_str, references=label_str, lang="it")

    f1 = []
    exact_match = []
    for ref, prediction in zip(val_answers, pred_str):
        exact_match.append(metric_max_over_ground_truths(exact_match_score, prediction, ref))
        f1.append(metric_max_over_ground_truths(f1_score, prediction, ref))
    exact_match = sum(exact_match) / len(exact_match)
    f1 = sum(f1) / len(f1)

    return {
        "R1": round(rouge_output["rouge1"], 4),
        "R2": round(rouge_output["rouge2"], 4),
        "RL": round(rouge_output["rougeL"], 4),
        "RLsum": round(rouge_output["rougeLsum"], 4),
        "BERT_P": round(sum(bertscore_output["precision"]) / len(bertscore_output["precision"]), 4),
        "BERT_R": round(sum(bertscore_output["recall"]) / len(bertscore_output["recall"]), 4),
        "BERT_F": round(sum(bertscore_output["f1"]) / len(bertscore_output["f1"]), 4),
        "EM": round(exact_match*100, 4),
        "F1": round(f1*100, 4),
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)



'''
----------------------------------------------------------------------------------------------------
                                        
                                        Defining Trainer

----------------------------------------------------------------------------------------------------
'''

trainer = transformers.Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()


'''
----------------------------------------------------------------------------------------------------

                                    Saving best model

----------------------------------------------------------------------------------------------------
'''

# Save the best model
trainer.save_model(args.checkpoint_dir + "/best_model")