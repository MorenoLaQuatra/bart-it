import transformers
import torch

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from data.pretraining_dataset import PretrainingDataset
from datasets import load_dataset
from time import time

# Initialize a BART-Base model
tokenizer = BartTokenizer.from_pretrained("tokenizer_bart_it")

"""
# Tiny version of BART
model = BartForConditionalGeneration(
    BartConfig(
        vocab_size=52000,
        max_position_embeddings=1024,
        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        d_model=512,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        is_encoder_decoder=True,
        decoder_start_token_id=tokenizer.eos_token_id,
    )
)
"""

streaming_dataset = load_dataset(
    "gsarti/clean_mc4_it", "full", split="train", streaming=True
)

dataset = PretrainingDataset(
    stream_dataset=streaming_dataset,
    tokenizer=tokenizer,
    max_input_length=1024,
    max_output_length=1024,
    padding="max_length",
    truncation=True,
    is_streaming=True,
)

t1 = time()
print(dataset[50000])
t2 = time()
print(dataset[500000])
t3 = time()
print(dataset[5000000])
t4 = time()

print("First time: ", t2 - t1)
print("Second time: ", t3 - t2)
print("Third time: ", t4 - t3)
