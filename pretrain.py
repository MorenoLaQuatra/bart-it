import transformers
import torch

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

# Initialize a BART-Base model
tokenizer = BartTokenizer.from_pretrained("tokenizer_bart_it")

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
