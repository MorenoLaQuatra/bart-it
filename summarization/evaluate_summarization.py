import transformers
from datasets import load_dataset
from summarization_dataset import Dataset
from torch.utils.data import DataLoader

import evaluate
import argparse
from tqdm import tqdm
import time


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

    parser.add_argument("--source_key", type=str, default="source", required=False)
    parser.add_argument("--target_key", type=str, default="summary", required=False)

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
# eval mode
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

try:
    test_data = load_dataset(args.dataset_name)["test"]
    test_input = test_data[args.source_key]
    test_target = test_data[args.target_key]
except Exception as e:
    print ("Error loading dataset splits. Most likely the dataset is not split into train, test and validation")
    print ("If you are using WITS, the default configuration randomly splits the dataset using 10K samples for test and 10K samples for validation and the rest for training")
    # take 10K samples for validation, 10K for testing, and the rest for training
    dataset = load_dataset(args.dataset_name)
    dataset = dataset["train"]
    dataset = dataset.shuffle(seed=42)
    test_dataset = dataset[10000:20000]

    test_input = test_dataset[args.source_key]
    test_target = test_dataset[args.target_key]


test_dataset = Dataset(source_text=test_input, target_text=test_target, tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

'''
----------------------------------------------------------------------------------------------------

                                    Evaluate Summarization Pipeline
                                    
----------------------------------------------------------------------------------------------------
'''

references = test_target
hypotheses = []
times = []


for batch in tqdm(test_dataloader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Generate Summary
    start_time = time.time()
    summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=args.max_length)
    end_time = time.time()
    times.append(end_time - start_time)

    # Decode Summary
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    '''
    print ("Summary: ", summary)
    print ("\n\n--------------------------------------------\n\n")
    '''


    # Append to list
    hypotheses.extend(summary)

if args.print_samples:
    for i in range(len(hypotheses)):
        print (f"Prediction: {hypotheses[i]}")
        print (f"Label: {references[i]}")
        print ("-----------------------\n\n")

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
rouge_results = rouge.compute(predictions=hypotheses, references=references, rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=False)
bertscore_output = bertscore.compute(predictions=hypotheses, references=references, lang="it")

print (f"Results for {args.model_path} on {args.dataset_name}")
print (f"ROUGE-1: {round(rouge_results['rouge1'] * 100, 2)}")
print (f"ROUGE-2: {round(rouge_results['rouge2'] * 100, 2)}")
print (f"ROUGE-L: {round(rouge_results['rougeL'] * 100, 2)}")
print (f"ROUGE-Lsum: {round(rouge_results['rougeLsum'] * 100, 2)}")
print (f'BERTScore P: {round((sum(bertscore_output["precision"]) / len(bertscore_output["precision"])) * 100, 2)}')
print (f'BERTScore R: {round((sum(bertscore_output["recall"]) / len(bertscore_output["recall"])) * 100, 2)}')
print (f'BERTScore F1: {round((sum(bertscore_output["f1"]) / len(bertscore_output["f1"])) * 100, 2)}')
print (f"Time per sample: {round(sum(times) / (len(times) * args.batch_size), 2)}")
