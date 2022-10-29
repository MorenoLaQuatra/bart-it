import transformers
from datasets import load_dataset
from qa_dataset import Dataset
from torch.utils.data import DataLoader

import evaluate
import argparse
from tqdm import tqdm

import re
from utils_qa import normalize_answer, normalize_answers, f1_score, exact_match_score, metric_max_over_ground_truths # from gsarti/it5
# ----------------------------------------------------------------------------------------------------
#
#                                         Parse Arguments
#
# ----------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", required=True)
    parser.add_argument("--tokenizer_path", type=str, default="", required=True)
    parser.add_argument("--dataset_name", type=str, default="", required=True)
    parser.add_argument("--max_length", type=int, default=128, required=False)
    parser.add_argument("--use_cuda", default=False, required=False, action="store_true")
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--print_samples", default=False, required=False, action="store_true")
    parser.add_argument("--max_input_length", type=int, default=1024, required=False)
    parser.add_argument("--max_output_length", type=int, default=128, required=False)
    parser.add_argument("--context_token", type=str, default="[CONTEXT]", required=False)
    parser.add_argument("--question", type=str, default="[QUESTION]", required=False)
    args = parser.parse_args()

    return args

args = parse_args()

'''
----------------------------------------------------------------------------------------------------

                                    Create Summarization model

----------------------------------------------------------------------------------------------------
'''

device = "cuda" if args.use_cuda else "cpu"

model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
if args.use_cuda:
    model = model.to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

test_data = load_dataset(args.dataset_name)["test"]
test_questions = test_data["question"]
test_context = test_data["context"]
test_answers = test_data["answers"]
test_answers = [e["text"] for e in test_answers]
single_answers = [ e[0] for e in test_answers]

test_dataset = Dataset(
    contexts=test_context,
    questions=test_questions,
    answers=single_answers,
    tokenizer=tokenizer,
    context_token=args.context_token,
    question_token=args.question,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
)

test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
'''
----------------------------------------------------------------------------------------------------

                                    Evaluate QA Pipeline
                                    
----------------------------------------------------------------------------------------------------
'''

references = test_answers
hypotheses = []

for batch in tqdm(test_dataloader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Generate Summary
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=args.max_length,
    )
    preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    hypotheses.extend(preds)

if args.print_samples:
    for i in range(len(hypotheses)):
        print("Question: ", test_questions[i])
        print("Context: ", test_context[i])
        print("Answer: ", test_answers[i])
        print("Predicted Answer: ", hypotheses[i])
        print("\n--------------------------------------------\n\n\n")

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
exact_match = evaluate.load("exact_match")
rouge_results = rouge.compute(predictions=hypotheses, references=references, rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=False)
bertscore_output = bertscore.compute(predictions=hypotheses, references=references, lang="it")
#exact_match_output = exact_match.compute(predictions=normalize_answers(list_of_answers=hypotheses), references=normalize_answers(list_of_answers=references))

f1 = []
exact_match = []
for prediction, ref in zip(hypotheses, references):
    exact_match.append(metric_max_over_ground_truths(exact_match_score, prediction, ref))
    f1.append(metric_max_over_ground_truths(f1_score, prediction, ref))
exact_match = sum(exact_match) / len(exact_match)
f1 = sum(f1) / len(f1)

print (f"Results for {args.model_path} on {args.dataset_name}")
print (f"ROUGE-1: {round(rouge_results['rouge1'] * 100, 2)}")
print (f"ROUGE-2: {round(rouge_results['rouge2'] * 100, 2)}")
print (f"ROUGE-L: {round(rouge_results['rougeL'] * 100, 2)}")
print (f"ROUGE-Lsum: {round(rouge_results['rougeLsum'] * 100, 2)}")
print (f'BERTScore P: {round((sum(bertscore_output["precision"]) / len(bertscore_output["precision"])) * 100, 2)}')
print (f'BERTScore R: {round((sum(bertscore_output["recall"]) / len(bertscore_output["recall"])) * 100, 2)}')
print (f'BERTScore F1: {round((sum(bertscore_output["f1"]) / len(bertscore_output["f1"])) * 100, 2)}')
#print (f'Exact Match HF : {round(exact_match_output["exact_match"] * 100, 2)}')
print (f'Exact Match IT5: {round(exact_match * 100, 2)}')
print (f'F1: {round(f1 * 100, 2)}')
