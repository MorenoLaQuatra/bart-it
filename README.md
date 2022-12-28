# BART-IT: Italian pretraining for BART sequence to sequence model

This repository contains the code for the pretraining BART-IT, an efficient and accurate sequence to sequence model for Italian language.

## Notes

As pointed out by the original author ([@gsarti_](https://twitter.com/gsarti_)) the IT5 model compared in the paper was not trained with multi-task learning, but with the regular span masking objective (as adopted from newer versions of T5).

## Table of Contents

- [Model Tokenizer](#model-tokenizer)
- [Model Pretraining](#model-pretraining)
- [Model Fine-Tuning](#model-fine-tuning)
- [Citation and acknowledgements](#citation-and-acknowledgements)

## Model Tokenizer

The code for training the tokenizer is self-contained in the `train_tokenizer.py` script. The tokenizer is trained on mC4, a large Italian corpus, and it is based on the BPE algorithm. The tokenizer is trained using the `tokenizers` library.

The following parameters are used to train the tokenizer:
- `vocab_size`: 52,000
- `min_frequency`: 10
- `special_tokens`: `<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>`

The tokenizer is saved in the `tokenizer_bart_it` folder.

## Model Pretraining

The main script for pretraining the model is `pretrain_base.py`. The model is trained following the same denoising pretraining strategy used for BART. Model parameters are reported on the table below.

| Parameter | Value |
| --- | --- |
| VOCAB_SIZE | 52,000 |
| MAX_POSITION_EMBEDDINGS | 1,024 |
| ENCODER_LAYERS | 6 |
| ENCODER_FFN_DIM | 3,072 |
| ENCODER_ATTENTION_HEADS | 12 |
| DECODER_LAYERS | 6 |
| DECODER_FFN_DIM | 3,072 |
| DECODER_ATTENTION_HEADS | 12 |
| D_MODEL | 768 |
| DROPOUT | 0.1 |

The model is trained on 2 NVIDIA RTX A6000 GPUs for a total of 1,7 million steps. The pre-trained model is released for the community on the [HuggingFace Hub](https://huggingface.co/) - [BART-IT](https://huggingface.co/morenolq/bart-it)

## Model Fine-tuning

The model is fine-tuned on the abstractive summarization task using the parameters reported in the table below.

| Parameter | Value |
| --- | --- |
| MAX_NUM_EPOCHS | 10 |
| BATCH_SIZE | 32 |
| LEARNING_RATE | 1e-5 |
| MAX_INPUT_LENGTH | 1024 |
| MAX_TARGET_LENGTH | 128 |

For more information about the model parameters, please refer to the `summarization/finetune_summarization.py` script and to the following [paper](https://doi.org/10.3390/fi15010015).

The model is fine-tuned on different summarization datasets and model weights for each dataset are released on the [HuggingFace Hub](https://huggingface.co/) - following table:

| Dataset Type | Dataset Name | Model Weights | Dataset Paper |
| --- | --- | --- | --- |
| News Summarization | FanPage | [`bart-it-fanpage`](https://huggingface.co/morenolq/bart-it-fanpage) | [Two New Datasets for Italian-Language Abstractive Text Summarization](https://doi.org/10.3390/info13050228) |
| News Summarization | IlPost | [`bart-it-ilpost`](https://huggingface.co/morenolq/bart-it-ilpost) | [Two New Datasets for Italian-Language Abstractive Text Summarization](https://doi.org/10.3390/info13050228) |
| Wikipedia Summarization | WITS | [`bart-it-WITS`](https://huggingface.co/morenolq/bart-it-WITS) | [WITS: Wikipedia for Italian Text Summarization](https://ceur-ws.org/Vol-3033/paper65.pdf) |

The model is an efficient and accurate sequence to sequence model for Italian language. The performance of the model are reported using both ROUGE and BERTScore metrics. Please refer to the following [paper](https://doi.org/10.3390/fi15010015) for more details.

The script for evaluating the model on the summarization task is `summarization/evaluate_summarization.py`.

## Citation and acknowledgments

If you use this code or the pre-trained model, please cite the following [paper](https://doi.org/10.3390/fi15010015):

```bibtex
@Article{BARTIT,
    AUTHOR = {La Quatra, Moreno and Cagliero, Luca},
    TITLE = {BART-IT: An Efficient Sequence-to-Sequence Model for Italian Text Summarization},
    JOURNAL = {Future Internet},
    VOLUME = {15},
    YEAR = {2023},
    NUMBER = {1},
    ARTICLE-NUMBER = {15},
    URL = {https://www.mdpi.com/1999-5903/15/1/15},
    ISSN = {1999-5903},
    DOI = {10.3390/fi15010015}
}
```

If you use the FanPage or IlPost datasets, please cite the following [paper](https://doi.org/10.3390/info13050228).

If you use the WITS dataset, please cite the following [paper](https://ceur-ws.org/Vol-3033/paper65.pdf).

If you use the mC4 dataset, please refer to the original [mT5 paper](https://arxiv.org/abs/2010.11934) and if you are interested to the cleaned version of the dataset, please refer to the [IT5 paper](https://arxiv.org/abs/2203.03759) and to the [cleaned mC4 repository](https://huggingface.co/datasets/gsarti/clean_mc4_it).
