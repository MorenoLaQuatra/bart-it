# --------------------------------------------------------------------
# This script is used to evaluate competitors on different datasets.
# --------------------------------------------------------------------

:'
# --------------------------------------------------------------------
# IlPost
# --------------------------------------------------------------------

# ARTeLab/ilpost - it5
python evaluate_summarization.py \
    --model_path "ARTeLab/it5-summarization-ilpost" \
    --tokenizer_path "ARTeLab/it5-summarization-ilpost" \
    --dataset_name "ARTeLab/ilpost" \
    --max_length 128 \
    --batch_size 4 \
    --use_cuda >> results/ilpost_test.log

# ARTeLab/ilpost - mbart
python evaluate_summarization.py \
    --model_path "ARTeLab/mbart-summarization-ilpost" \
    --tokenizer_path "ARTeLab/mbart-summarization-ilpost" \
    --dataset_name "ARTeLab/ilpost" \
    --max_length 128 \
    --batch_size 4 \
    --use_cuda >> results/ilpost_test.log



# ARTeLab/ilpost - google/mt5-base
python evaluate_summarization.py \
    --model_path "checkpoints/ilpost_mt5/checkpoint-4402/" \
    --tokenizer_path "google/mt5-base" \
    --dataset_name "ARTeLab/ilpost" \
    --max_length 128 \
    --batch_size 2 \
    --target_key "target" \
    --use_cuda >> results/ilpost_test.log

'


:'
# --------------------------------------------------------------------
# Fanpage - News Summarization
# --------------------------------------------------------------------

# ARTeLab/fanpage - it5
python evaluate_summarization.py \
    --model_path "ARTeLab/it5-summarization-fanpage" \
    --tokenizer_path "ARTeLab/it5-summarization-fanpage" \
    --dataset_name "ARTeLab/fanpage" \
    --max_length 128 \
    --batch_size 4 \
    --use_cuda >> results/fanpage_test.log

# ARTeLab/fanpage - mbart
python evaluate_summarization.py \
    --model_path "ARTeLab/mbart-summarization-fanpage" \
    --tokenizer_path "ARTeLab/mbart-summarization-fanpage" \
    --dataset_name "ARTeLab/fanpage" \
    --max_length 128 \
    --batch_size 4 \
    --use_cuda >> results/fanpage_test.log

# ARTeLab/fanpage - it5/mt5-base-news-summarization
python evaluate_summarization.py \
    --model_path "it5/mt5-base-news-summarization" \
    --tokenizer_path "it5/mt5-base-news-summarization" \
    --dataset_name "ARTeLab/fanpage" \
    --max_length 128 \
    --batch_size 16 \
    --target_key "target" \
    --use_cuda >> results/fanpage_test.log
'

:'
# --------------------------------------------------------------------
# MLSum
# --------------------------------------------------------------------


# ARTeLab/mlsum-it - it5
python evaluate_summarization.py \
    --model_path "ARTeLab/it5-summarization-mlsum" \
    --tokenizer_path "ARTeLab/it5-summarization-mlsum" \
    --dataset_name "ARTeLab/mlsum-it" \
    --max_length 128 \
    --batch_size 4 \
    --use_cuda >> results/mlsum_test.log

# ARTeLab/mlsum-it - mbart
python evaluate_summarization.py \
    --model_path "ARTeLab/mbart-summarization-mlsum" \
    --tokenizer_path "ARTeLab/mbart-summarization-mlsum" \
    --dataset_name "ARTeLab/mlsum-it" \
    --max_length 128 \
    --batch_size 4 \
    --use_cuda >> results/mlsum_test.log

'

# --------------------------------------------------------------------
# WITS
# --------------------------------------------------------------------

# Silvia/WITS - it5

python evaluate_summarization.py \
    --model_path "checkpoints/wits_it5/best_model" \
    --tokenizer_path "gsarti/it5-base" \
    --dataset_name "Silvia/WITS" \
    --max_length 128 \
    --batch_size 4 \
    --use_cuda >> results/wits_test.log


# Silvia/WITS - mbart

python evaluate_summarization.py \
    --model_path "checkpoints/wits_mbart/checkpoint-28310/" \
    --tokenizer_path "facebook/mbart-large-cc25" \
    --dataset_name "Silvia/WITS" \
    --max_length 128 \
    --batch_size 4 \
    --print_samples >> results/wits_test.log

# Silvia/WITS - google/mt5-base

python evaluate_summarization.py \
    --model_path "checkpoints/wits_mt5/best_model/" \
    --tokenizer_path "google/mt5-base" \
    --dataset_name "Silvia/WITS" \
    --max_length 128 \
    --batch_size 2 \
    --use_cuda >> results/wits_test.log

