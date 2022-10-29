# --------------------------------------------------------------------
# This script is used to fine-tune the model on different datasets.
# --------------------------------------------------------------------

export COMET_API_KEY="HN6FbVyLBsdD2ghHbKzTPseHA"
export COMET_PROJECT_NAME="bart_it_ilpost"
export COMET_WORKSPACE="morenolaquatra"

# ARTeLab/ilpost 
python finetune_summarization.py \
    --model_path "../bart-it-s" \
    --log_dir "ilpost_logs" \
    --checkpoint_dir "ilpost_checkpoints" \
    --dataset_name "ARTeLab/ilpost" \
    --tokenizer_path "../tokenizer_bart_it" \
    --use_cuda \
    --fp16 > ilpost_finetune.log

# ARTeLab/fanpage

export COMET_API_KEY="HN6FbVyLBsdD2ghHbKzTPseHA"
export COMET_PROJECT_NAME="bart_it_fanpage"
export COMET_WORKSPACE="morenolaquatra"

python finetune_summarization.py \
    --model_path "../bart-it-s" \
    --log_dir "fanpage_logs" \
    --checkpoint_dir "fanpage_checkpoints" \
    --dataset_name "ARTeLab/fanpage" \
    --tokenizer_path "../tokenizer_bart_it" \
    --use_cuda \
    --fp16 > fanpage_finetune.log

# ARTeLab/mlsum-it

export COMET_API_KEY="HN6FbVyLBsdD2ghHbKzTPseHA"
export COMET_PROJECT_NAME="bart_it_mlsum"
export COMET_WORKSPACE="morenolaquatra"

python finetune_summarization.py \
    --model_path "../bart-it-s" \
    --log_dir "mlsum_logs" \
    --checkpoint_dir "mlsum_checkpoints" \
    --dataset_name "ARTeLab/mlsum-it" \
    --tokenizer_path "../tokenizer_bart_it" \
    --use_cuda \
    --fp16 > mlsum_finetune.log