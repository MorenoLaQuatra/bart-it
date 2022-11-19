from typing import List

import torch
import transformers


class Dataset(torch.utils.data.Dataset):
    """
    This class is inteded for sequence classification tasks.
    :param source_text: List of source text that is used as input to the model.
    :param target_text: List of target text that is used as expected output from the model.
    :param tokenizer: The tokenizer to be used for tokenizing the texts. It can be an instance of the transformers AutoTokenizer class or a custom tokenizer.
    :param max_input_length: The maximum length of the tokenized input text.
    :param max_output_length: The maximum length of the tokenized output text.
    :param padding: The padding strategy to be used. Available options are available in the transformers library.
    :param truncation: Whether to truncate the text or not.
    """

    def __init__(
        self,
        source_text: List[str],
        target_text: List[str],
        tokenizer,
        max_input_length: int = 1024,
        max_output_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
    ):

        self.source_text = source_text
        self.target_text = target_text
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation

    def __getitem__(self, idx):
        """
        This function is called to get the tokenized source and target text for a given index.
        :param idx: The index of the text and label to be returned.
        :return: A dictionary containing the tokenized source (`input_ids`) with attention mask (`attention_mask`) and the tokenized target (`labels`).
        """
        input = self.tokenizer(
            self.source_text[idx],
            max_length=self.max_input_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )
        try:
            output = self.tokenizer(
                text_target = self.target_text[idx],
                max_length=self.max_output_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors="pt",
            )
        except Exception as e:
            #print (e)
            output = self.tokenizer(
                self.target_text[idx],
                max_length=self.max_output_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors="pt",
            )

        labels = output["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        item = {
            "input_ids": input["input_ids"].squeeze(),
            "attention_mask": input["attention_mask"].squeeze(),
            "labels": labels,
            #"labels_mask": output["attention_mask"].squeeze(),
        }

        return item

    def __len__(self):
        return len(self.source_text)
