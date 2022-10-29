from data.utils import (
    sentence_permutation,
    document_rotation,
    text_infilling,
    token_masking,
    token_deletion,
)
from transformers import AutoTokenizer

text = """
Questa è una frase lunga che verrà usata per fare dei test.
Le parole sono scritte in modo casuale per simulare un testo reale.
In questo modo si possono testare le funzioni di perturbazione.
Le funzioni di perturbazione sono state implementate in modo da poter essere usate in modo indipendente.
"""
text = text.replace("\n", " ")

tokenizer = AutoTokenizer.from_pretrained("tokenizer_bart_it")
list_special_tokens = tokenizer.all_special_ids

# sentence_permutation - operates on text strings
perturbed_text = sentence_permutation(text)
print("\n\nText:", text)
print("PERTURBED sentence_permutation:", perturbed_text)

perturbed_text = document_rotation(text)
print("\n\nText:", text)
print("PERTURBED document_rotation:", perturbed_text)

tokenized_input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
perturbed_tokenized_text = token_deletion(
    tokenized_input_ids, list_special_tokens=list_special_tokens
)
perturbed_text = tokenizer.decode(perturbed_tokenized_text)
print("\n\nText:", text)
print("PERTURBED token_deletion:", perturbed_text)

tokenized_input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
perturbed_tokenized_text = token_masking(
    tokenized_input_ids,
    mask_token_id=tokenizer.mask_token_id,
    list_special_tokens=list_special_tokens,
)
perturbed_text = tokenizer.decode(perturbed_tokenized_text)
print("\n\nText:", text)
print("PERTURBED token_masking:", perturbed_text)

tokenized_input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
perturbed_tokenized_text = token_infilling(
    tokenized_input_ids,
    mask_token_id=tokenizer.mask_token_id,
    list_special_tokens=list_special_tokens,
)
perturbed_text = tokenizer.decode(perturbed_tokenized_text)
print("\n\nText:", text)
print("PERTURBED token_infilling:", perturbed_text)
