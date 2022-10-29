import random
import torch


def sentence_permutation(text: str) -> str:
    """
    A document is divided into sentences based on full stops, and these sentences are shuffled in a random order.
    **This function operates on text strings.**
    :param sentence: The sentence to be permuted.
    :return: The permuted sentence.
    """
    sentences = text.split(".")
    permuted_sentences = torch.randperm(len(sentences))
    permuted_text = ""
    for i in permuted_sentences:
        if sentences[i] != "":
            permuted_text += sentences[i] + ". "
    return permuted_text.strip()


def token_infilling(
    tokenized_sequence: torch.Tensor,
    mask_token_id: int,
    mask_probability: float = 0.15,
    list_special_tokens: list = [],
) -> str:
    """
    A number of text spans are sampled, with span lengths drawn from a Poisson distribution (λ = 3).
    Each span is replaced with a single [MASK] token. 0-length spans correspond to the insertion of
    [MASK] tokens. Text infilling is inspired by SpanBERT (Joshi et al., 2019), but SpanBERT samples
    span lengths from a different (clamped geometric) distribution, and replaces each span with a
    sequence of [MASK] tokens of exactly the same length. Text infilling teaches the model to predict
    how many tokens are missing from a span.
    **This function operates on tokenized text.**
    :param text: The text to be infilled.
    :return: The infilled text.
    """
    span_length = int(torch.poisson(torch.tensor([3.0])))
    perturbed_ids = torch.empty(0, dtype=torch.long)
    if span_length > 0:
        for i in range(0, len(tokenized_sequence), span_length):
            if torch.rand(1) < mask_probability:
                # check if the span does not contain special tokens
                if not any(token in list_special_tokens for token in tokenized_sequence[i : i + span_length]):
                    perturbed_ids = torch.cat(
                        (perturbed_ids, torch.tensor([mask_token_id], dtype=torch.long))
                    )
            else:
                perturbed_ids = torch.cat(
                    (perturbed_ids, tokenized_sequence[i : i + span_length])
                )
    else:
        perturbed_ids = tokenized_sequence # if the span length is 0, the text is not perturbed
    return perturbed_ids


def token_masking(
    tokenized_sequence: torch.Tensor,
    mask_token_id: int,
    mask_probability: float = 0.15,
    list_special_tokens: list = [],
) -> str:
    """
    Random tokens are replaced with the [MASK] token. This task trains the model to predict the original value of the masked tokens.
    **This function operates on tokenized text.**
    :param text: The text to be masked.
    :return: The masked text.
    """
    for i in range(len(tokenized_sequence)):
        if torch.rand(1) < mask_probability:
            if tokenized_sequence[i] not in list_special_tokens:
                tokenized_sequence[i] = mask_token_id
    return tokenized_sequence


def token_deletion(
    tokenized_sequence: torch.Tensor,
    mask_token_id: int,
    mask_probability: float = 0.15,
    list_special_tokens: list = [],
) -> str:
    """
    Random tokens are deleted from the input. In contrast to token masking, the model must decide which positions are missing inputs.
    **This function operates on tokenized text.**
    :param text: The text to be token deleted.
    :return: The token deleted text.
    """
    delete_mask = torch.rand(len(tokenized_sequence)) < mask_probability
    tokenized_sequence = tokenized_sequence[~delete_mask]
    return tokenized_sequence


def document_rotation(text: str) -> str:
    """
    A token is chosen uniformly at random, and the document is rotated so that it begins with that token.
    This task trains the model to identify the start of the document.
    **This function operates on text strings.**
    :param text: The text to be rotated.
    :return: The rotated text.
    """
    text = text.split(" ")
    rotation_index = random.randint(0, len(text) - 1)
    rotated_text = text[rotation_index:] + text[:rotation_index]
    return " ".join(rotated_text)
