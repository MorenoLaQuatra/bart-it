import comet_ml
import transformers
import torch
import os

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from datasets import load_dataset
from time import time
from data.utils import sentence_permutation, document_rotation
from data.utils import token_infilling, token_masking, token_deletion

import random


comet_ml.init(project_name='bart-it-base')
# 1. Enable logging of model checkpoints
os.environ["COMET_LOG_ASSETS"] = "True"

# PARAMETERS BART BASE
# ==============================================================================
VOCAB_SIZE = 52000
MAX_POSITION_EMBEDDINGS = 1024
ENCODER_LAYERS = 6
ENCODER_FFN_DIM = 3072
ENCODER_ATTENTION_HEADS = 12
DECODER_LAYERS = 6
DECODER_FFN_DIM = 3072
DECODER_ATTENTION_HEADS = 12
D_MODEL = 768
DROPOUT = 0.1
# ==============================================================================
# PARAMETERS 



# Initialize a BART-Base model
tokenizer = BartTokenizer.from_pretrained("tokenizer_bart_it")


# Tiny version of BART
model = BartForConditionalGeneration(
    BartConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        encoder_layers=ENCODER_LAYERS,
        encoder_ffn_dim=ENCODER_FFN_DIM,
        encoder_attention_heads=ENCODER_ATTENTION_HEADS,
        decoder_layers=DECODER_LAYERS,
        decoder_ffn_dim=DECODER_FFN_DIM,
        decoder_attention_heads=DECODER_ATTENTION_HEADS,
        d_model=D_MODEL,
        dropout=DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        is_encoder_decoder=True,
        decoder_start_token_id=tokenizer.eos_token_id,
    )
)


train_streaming_dataset = load_dataset(
    "gsarti/clean_mc4_it", "full", split="train", streaming=True
).with_format(type="torch")
eval_streaming_dataset = load_dataset(
    "gsarti/clean_mc4_it", "full", split="validation", streaming=True
).with_format(type="torch")



# perturbation in string: document_rotation, sentence_permutation
# perturbation in token : token_infilling, token_masking, token_deletion
perturbations = [
    document_rotation,
    sentence_permutation,
    token_infilling,
    token_masking,
    token_deletion,
]

perturbations_text_domain = [
    document_rotation,
    sentence_permutation,
]

perturbations_token_domain = [
    token_infilling,
    token_masking,
    token_deletion,
]


def collate_fn(examples):
    """
    Collate function to be used in the dataloader.
    It applies the perturbations to the examples and returns the batch.
    TODO: improve efficiency
    :param examples: list of examples
    :return: batch ready to be fed to the model
    """
    original_texts = [example["text"] for example in examples]

    input_ids = None
    for text in original_texts:
        perturbation_function = random.choice(perturbations)
        if perturbation_function in perturbations_text_domain:
            # need to truncate the text to 1024 tokens
            t_text = tokenizer(text, truncation=True, max_length=1024)
            text_truncated = tokenizer.decode(t_text["input_ids"], skip_special_tokens=True)
            perturbed_text = perturbation_function(text_truncated)
            perturbed_input_ids = tokenizer(
                perturbed_text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_POSITION_EMBEDDINGS
            )["input_ids"][0]
        else:
            original_input_ids = tokenizer(
                text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_POSITION_EMBEDDINGS
            )["input_ids"][0]
            perturbed_input_ids = perturbation_function(
                    tokenized_sequence=original_input_ids,
                    mask_token_id=tokenizer.mask_token_id,
                    mask_probability=0.15,
                    list_special_tokens=tokenizer.all_special_ids,
                )
            if perturbed_input_ids.shape[-1] < MAX_POSITION_EMBEDDINGS: # apply padding
                perturbed_input_ids = torch.cat(
                    (perturbed_input_ids, torch.full((MAX_POSITION_EMBEDDINGS - perturbed_input_ids.shape[-1],),
                    tokenizer.pad_token_id, 
                    dtype=torch.long)))
            perturbed_input_ids = torch.squeeze(perturbed_input_ids, dim=0)
            
        if input_ids is None:
            input_ids = perturbed_input_ids.unsqueeze(0)
        else:
            input_ids = torch.cat((input_ids, perturbed_input_ids.unsqueeze(0)), dim=0)

    tokenized_examples = {}
    # update the tokenized examples with the perturbed input ids and convert to tensors
    tokenized_examples["input_ids"] = input_ids
    # update the attention mask
    tokenized_examples["attention_mask"] = [
        [1 if token_id != tokenizer.pad_token_id else 0 for token_id in input_ids]
        for input_ids in tokenized_examples["input_ids"]
    ]
    tokenized_examples["attention_mask"] = torch.tensor(tokenized_examples["attention_mask"])
    
    tokenized_examples["labels"] = tokenizer(
        original_texts, padding="max_length", truncation=True, max_length=MAX_POSITION_EMBEDDINGS, return_tensors="pt"
    )["input_ids"]

    return tokenized_examples


# total_steps (1 epoch, see it5) = 103_000_000 / 64 = 1_609_375 -- 1_700_000
# warmup_steps = 1_700_000 * 0.01 = 17_000

# Prepare training arguments
training_args = transformers.TrainingArguments(
    output_dir="./bart-it-size-s",
    overwrite_output_dir=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    warmup_steps=17_000,
    weight_decay=0.01,
    save_strategy="steps",
    evaluation_strategy="steps",
    max_steps=1_700_000,
    logging_dir="./logs-bart-it-size-s",
    logging_steps=100,
    eval_steps=10000,
    save_steps=10000,
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    fp16=True,
    dataloader_num_workers=24,
    learning_rate=1e-4,
)

# Initialize the trainer

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_streaming_dataset,
    eval_dataset=eval_streaming_dataset,
    data_collator=collate_fn,
)

# Train the model
trainer.train()

# Evaluate the model
print(trainer.evaluate(eval_streaming_dataset))

# Save the model
trainer.save_model("./bart-it-s")





