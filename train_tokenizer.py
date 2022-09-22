"""
This script is used to train a tokenizer on clean_mc4_it dataset.
It trains a Byte-Pair Encoding (BPE) tokenizer using the tokenizers library by HuggingFace.
The tokenizer is stored using a format that is compatible with the transformers library.
"""
import os

from transformers import AutoTokenizer, BartTokenizer
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Load the dataset to train the tokenizer on
dataset = load_dataset("gsarti/clean_mc4_it", "full", split="train", streaming=True)

# shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000) # shuffle the dataset is not required for the training of the tokenizer

# define iterator function to yield the text from the dataset
def iterator():
    for example in dataset:
        yield example["text"]


# Train the tokenizer using the dataset and the parameters provided
tokenizer.train_from_iterator(
    iterator(),
    vocab_size=52_000,
    min_frequency=10,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
    show_progress=True,
)

cls_token_id = tokenizer.token_to_id("<s>")
sep_token_id = tokenizer.token_to_id("</s>")
print(cls_token_id, sep_token_id)


# save the tokenizer
if not os.path.exists("tokenizer"):
    os.mkdir("tokenizer")
tokenizer.model.save("tokenizer")

# Initialize a tokenizer : this is the one that will be used for the training
# this trick is needed to comply with the HuggingFace API
tokenizer = BartTokenizer(
    vocab_file="tokenizer/vocab.json", merges_file="tokenizer/merges.txt"
)
tokenizer.save_pretrained("tokenizer_bart_it")
tokenizer = AutoTokenizer.from_pretrained("tokenizer_bart_it")

# clean folder that contains the old tokenizer
os.system("rm -rf tokenizer")

# Example of encoding a text
encoded_input = tokenizer("Ciao, come stai?", return_tensors="pt")
print(encoded_input)
decoded = tokenizer.decode(encoded_input["input_ids"][0], skip_special_tokens=True)
print(decoded)
