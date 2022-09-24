from typing import List

import torch
import transformers
import datasets
import random


class PretrainingDataset(torch.utils.data.Dataset):
    """
    This class is inteded for sequence classification tasks.
    :param source_text: List of source text that is used as input to the model.
    :param stream_dataset: IterableDataset that is used as input to the model.
    :param tokenizer: The tokenizer to be used for tokenizing the texts. It can be an instance of the transformers AutoTokenizer class or a custom tokenizer.
    :param max_input_length: The maximum length of the tokenized input text.
    :param max_output_length: The maximum length of the tokenized output text.
    :param padding: The padding strategy to be used. Available options are available in the transformers library.
    :param truncation: Whether to truncate the text or not.
    :param is_streaming: Whether the dataset is a stream dataset or not.
    """

    def __init__(
        self,
        source_text: List[str] = None,
        stream_dataset: datasets.IterableDataset = None,
        tokenizer: transformers.PreTrainedTokenizer = None,
        max_input_length: int = 1024,
        max_output_length: int = 1024,
        padding: str = "max_length",
        truncation: bool = True,
        is_streaming: bool = False,
    ):

        if is_streaming:
            self._check_none(
                [
                    ("source_text", source_text),
                ]
            )

            self._check_not_none(
                [
                    ("tokenizer", tokenizer),
                    ("stream_dataset", stream_dataset),
                ]
            )
            self.stream_dataset = stream_dataset
        else:
            self._check_none(
                [
                    ("stream_dataset", stream_dataset),
                ]
            )
            self._check_not_none(
                [
                    ("tokenizer", tokenizer),
                    ("source_text", source_text),
                ]
            )
            self.source_text = source_text

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.is_streaming = is_streaming

    def _check_not_none(self, list_values: List):
        """
        This function checks if any of the arguments is None.
        :param args: The arguments to be checked.
        """
        for arg_name, arg in list_values:
            if arg is None:
                raise ValueError(f"The argument {arg_name} cannot be None.")

    def _check_none(self, list_values: List):
        """
        This function checks if any of the arguments is not None.
        :param args: The arguments to be checked.
        """
        for arg_name, arg in list_values:
            if arg is not None:
                raise ValueError(f"The argument {arg_name} must be None.")

    def sentence_permutation(self, sentence: str) -> str:
        """
        A document is divided into sentences based on full stops, and these sentences are shuffled in a random order.
        :param sentence: The sentence to be permuted.
        :return: The permuted sentence.
        """
        sentences = sentence.split(".")
        permuted_sentences = torch.randperm(len(sentences))
        permuted_sentence = ""
        for i in permuted_sentences:
            permuted_sentence += sentences[i] + "."
        return permuted_sentence

    def text_infilling(self, text: str) -> str:
        """
        A number of text spans are sampled, with span lengths drawn from a Poisson distribution (λ = 3).
        Each span is replaced with a single [MASK] token. 0-length spans correspond to the insertion of
        [MASK] tokens. Text infilling is inspired by SpanBERT (Joshi et al., 2019), but SpanBERT samples
        span lengths from a different (clamped geometric) distribution, and replaces each span with a
        sequence of [MASK] tokens of exactly the same length. Text infilling teaches the model to predict
        how many tokens are missing from a span
        :param text: The text to be infilled.
        :return: The infilled text.
        """
        text = text.split(" ")
        text_length = len(text)
        infilled_text = ""
        for i in range(text_length):
            if torch.rand(1) < 0.15:
                if torch.rand(1) < 0.8:
                    infilled_text += "[MASK] "
                else:
                    if torch.rand(1) < 0.5:
                        infilled_text += text[i] + " "
                    else:
                        infilled_text += "[MASK] "
            else:
                infilled_text += text[i] + " "
        return infilled_text

    def token_masking(self, text: str) -> str:
        """
        Random tokens are replaced with the [MASK] token. This task trains the model to predict the original value of the masked tokens.
        :param text: The text to be masked.
        :return: The masked text.
        """
        text = text.split(" ")
        text_length = len(text)
        masked_text = ""
        for i in range(text_length):
            if torch.rand(1) < 0.15:
                masked_text += "[MASK] "
            else:
                masked_text += text[i] + " "
        return masked_text

    def token_deletion(self, text: str) -> str:
        """
        Random tokens are deleted from the input. In contrast to token masking,
        the model must decide which positions are missing inputs.
        :param text: The text to be token deleted.
        :return: The token deleted text.
        """
        text = text.split(" ")
        text_length = len(text)
        deleted_text = ""
        for i in range(text_length):
            if torch.rand(1) < 0.15:
                deleted_text += ""
            else:
                deleted_text += text[i] + " "
        return deleted_text

    def document_rotation(self, document: str) -> str:
        """
        A token is chosen uniformly at random, and the document is rotated so that it begins with that token.
        This task trains the model to identify the start of the document.
        :param document: The document to be rotated.
        :return: The rotated document.
        """
        document = document.split(" ")
        document_length = len(document)
        random_index = torch.randint(0, document_length, (1,)).item()
        rotated_document = ""
        for i in range(document_length):
            rotated_document += document[(i + random_index) % document_length] + " "
        return rotated_document

    def __getitem__(self, idx):
        """
        This function is called to get the tokenized source and target text for a given index.
        :param idx: The index of the text and label to be returned.
        :return: A dictionary containing the tokenized source (`input_ids`) with attention mask (`attention_mask`) and the tokenized target (`labels`).
        """
        if self.is_streaming:
            text = list(self.stream_dataset.skip(idx).take(1))[0]["text"]
        else:
            text = self.source_text[idx]

        print("TEXT: ", text)

        # this function apply perturbation to the text that is passed to the model as input
        perturbation_function = random.choice(
            [
                self.sentence_permutation,
                self.text_infilling,
                self.token_masking,
                self.token_deletion,
                self.document_rotation,
            ]
        )
        print("\n\n*********\nPERTURBATION FUNCTION: ", perturbation_function)

        input_text = perturbation_function(text)

        print("INPUT TEXT: ", input_text)

        # the input of the model is the perturbed text
        input = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        # the output of the model is the correct text that is passed to the model as target
        output = self.tokenizer(
            text_target=text,
            max_length=self.max_output_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        item = {
            "input_ids": input["input_ids"].squeeze(),
            "attention_mask": input["attention_mask"].squeeze(),
            "labels": output["input_ids"].squeeze(),
        }

        return item

    def __len__(self):
        """
        This function is called to get the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.source_text)
