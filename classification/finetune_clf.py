import comet_ml
import transformers
from datasets import load_dataset
import argparse
from summarization_dataset import Dataset
import evaluate

import torch


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
    parser.add_argument("--learning_rate", type=float, default=1e-4, required=False)
    parser.add_argument("--dataloader_num_workers", type=int, default=8, required=False)
    parser.add_argument("--save_total_limit", type=int, default=5, required=False)
    parser.add_argument("--use_cuda", default=False, action="store_true", required=False)
    parser.add_argument("--fp16", default=False, action="store_true", required=False)
    parser.add_argument("--hub_model_id", type=str, default="", required=False)
    parser.add_argument("--push_to_hub", default=False, action="store_true", required=False)

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

model = transformers.AutoModelForSequenceClassification.from_pretrained(args.model_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

dataset = load_dataset(args.dataset_name)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
val_dataset = dataset["validation"]

print ("Len train: ", len(train_dataset))
print ("Len test: ", len(test_dataset))
print ("Len val: ", len(val_dataset))

'''
# average length input in training set
avg_len_train = sum([len(tokenizer.encode(x)) for x in train_dataset["source"]]) / len(train_dataset)
# average length output in training set
avg_len_output_train = sum([len(tokenizer.encode(x)) for x in train_dataset["target"]]) / len(train_dataset)

# average length input in test set
avg_len_test = sum([len(tokenizer.encode(x)) for x in test_dataset["source"]]) / len(test_dataset)
# average length output in test set
avg_len_output_test = sum([len(tokenizer.encode(x)) for x in test_dataset["target"]]) / len(test_dataset)

# average length input in validation set
avg_len_val = sum([len(tokenizer.encode(x)) for x in val_dataset["source"]]) / len(val_dataset)
# average length output in validation set
avg_len_output_val = sum([len(tokenizer.encode(x)) for x in val_dataset["target"]]) / len(val_dataset)
'''

train_input = train_dataset["source"]
train_target = train_dataset["target"]

test_input = test_dataset["source"]
test_target = test_dataset["target"]

val_input = val_dataset["source"]
val_target = val_dataset["target"]


train_dataset = Dataset(source_text=train_input, target_text=train_target, tokenizer=tokenizer, max_output_length=args.max_output_length, max_input_length=args.max_input_length)
#test_dataset = Dataset(source_text=test_input, target_text=test_target, tokenizer=tokenizer, max_output_length=args.max_output_length, max_input_length=args.max_input_length)
val_dataset = Dataset(source_text=val_input, target_text=val_target, tokenizer=tokenizer, max_output_length=args.max_output_length, max_input_length=args.max_input_length)


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
    metric_for_best_model="R2",
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

def compute_metrics(pred):

    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = val_target
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


    return {
        "R1": round(rouge_output["rouge1"], 4),
        "R2": round(rouge_output["rouge2"], 4),
        "RL": round(rouge_output["rougeL"], 4),
        "RLsum": round(rouge_output["rougeLsum"], 4),
        "BERT_P": round(sum(bertscore_output["precision"]) / len(bertscore_output["precision"]), 4),
        "BERT_R": round(sum(bertscore_output["recall"]) / len(bertscore_output["recall"]), 4),
        "BERT_F": round(sum(bertscore_output["f1"]) / len(bertscore_output["f1"]), 4),
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

                        Evaluating on test set and saving best model

----------------------------------------------------------------------------------------------------
'''
print (f"Test set evaluation {args.dataset_name}: {trainer.evaluate(test_dataset)}")

# Save the best model
trainer.save_model(args.checkpoint_dir + "/best_model")