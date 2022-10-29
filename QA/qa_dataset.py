from typing import List

import torch
import transformers


class Dataset(torch.utils.data.Dataset):
    """
    This class is inteded for sequence classification tasks.
    :param contexts: List of source contexts that is used as input to the model.
    :param questions: List of source questions that is used as input to the model.
    :param answers: List of target answers that is used as expected output from the model.
    :param tokenizer: The tokenizer to be used for tokenizing the texts. It can be an instance of the transformers AutoTokenizer class or a custom tokenizer.
    :param context_token: The token to be used to separate the context from the question.
    :param question_token: The token to be used to separate the question from the context.
    :param max_input_length: The maximum length of the tokenized input text.
    :param max_output_length: The maximum length of the tokenized output text.
    :param padding: The padding strategy to be used. Available options are available in the transformers library.
    :param truncation: Whether to truncate the text or not.
    """

    def __init__(
        self,
        contexts: List[str],
        questions: List[str],
        answers: List[str],
        tokenizer,
        context_token: str = "[CONTEXT]",
        question_token: str = "[QUESTION]",
        max_input_length: int = 256,
        max_output_length: int = 64,
        padding: str = "max_length",
        truncation: bool = True,
    ):

        self.contexts = contexts
        self.questions = questions
        self.answers = answers

        self.tokenizer = tokenizer
        self.context_token = context_token
        self.question_token = question_token

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
            text=self.context_token + self.contexts[idx] + self.question_token + self.questions[idx],
            max_length=self.max_input_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        try:
            output = self.tokenizer(
                text_target = self.answers[idx],
                max_length=self.max_output_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors="pt",
            )
        except Exception as e:
            output = self.tokenizer(
                self.answers[idx],
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
        return len(self.contexts)
