# --------------------------------------------------------------------
# IlPost
# --------------------------------------------------------------------

# ARTeLab/ilpost - google/mt5-base
:'
export COMET_PROJECT_NAME="bart_it_ilpost"

python finetune_summarization.py \
    --model_path "google/mt5-base" \
    --tokenizer_path "google/mt5-base" \
    --log_dir "logs/ilpost_mt5" \
    --checkpoint_dir "checkpoints/ilpost_mt5/" \
    --dataset_name "ARTeLab/ilpost" \
    --use_cuda \
    --save_total_limit 2 \
    --batch_size 8 \
    --target_key "target" \
    --epochs 5 > logs/ilpost_mt5_finetune.log


# --------------------------------------------------------------------
# WITS
# --------------------------------------------------------------------

# Silvia/WITS - gsarti/it5-base

export COMET_PROJECT_NAME="bart_it_wits"

python finetune_summarization.py \
    --model_path "gsarti/it5-base" \
    --tokenizer_path "gsarti/it5-base" \
    --log_dir "logs/wits_it5" \
    --checkpoint_dir "checkpoints/wits_it5/" \
    --dataset_name "Silvia/WITS" \
    --target_key "summary" \
    --use_cuda \
    --save_total_limit 2 \
    --batch_size 8 \
    --target_key "summary" \
    --epochs 5 > logs/wits_it5_finetune.log
'

# Silvia/WITS - facebook/mbart-large-cc25

python finetune_summarization.py \
    --model_path "facebook/mbart-large-cc25" \
    --tokenizer_path "facebook/mbart-large-cc25" \
    --log_dir "logs/wits_mbart" \
    --checkpoint_dir "checkpoints/wits_mbart/" \
    --dataset_name "Silvia/WITS" \
    --target_key "summary" \
    --use_cuda \
    --save_total_limit 2 \
    --batch_size 12 \
    --epochs 5 > logs/wits_mbart_finetune.log

:'
# Silvia/WITS - google/mt5-base

python finetune_summarization.py \
    --model_path "google/mt5-base" \
    --tokenizer_path "google/mt5-base" \
    --log_dir "logs/wits_mt5" \
    --checkpoint_dir "checkpoints/wits_mt5/" \
    --dataset_name "Silvia/WITS" \
    --target_key "summary" \
    --use_cuda \
    --save_total_limit 2 \
    --batch_size 8 \
    --target_key "summary" \
    --epochs 5 > logs/wits_mt5_finetune.log

'
# --------------------------------------------------------------------
# FanPage
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# MLSum
# --------------------------------------------------------------------

:'
# ARTeLab/mlsum-it - google/mt5-base

export COMET_PROJECT_NAME="bart_it_mlsum"

python finetune_summarization.py \
    --model_path "google/mt5-base" \
    --tokenizer_path "google/mt5-base" \
    --log_dir "logs/mlsum_mt5" \
    --checkpoint_dir "checkpoints/mlsum_mt5/" \
    --dataset_name "ARTeLab/mlsum-it" \
    --use_cuda \
    --save_total_limit 2 \
    --batch_size 8 \
    --target_key "target" \
    --epochs 5 > logs/mlsum_mt5_finetune.log
'