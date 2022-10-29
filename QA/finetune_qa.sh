export COMET_API_KEY="HN6FbVyLBsdD2ghHbKzTPseHA"
export COMET_PROJECT_NAME="squad_it_s2s"
export COMET_WORKSPACE="morenolaquatra"

# squad_it
python finetune_qa.py \
    --model_path "../bart-it-s" \
    --log_dir "bart_squad_it" \
    --checkpoint_dir "bart_squad_it" \
    --dataset_name "squad_it" \
    --tokenizer_path "../tokenizer_bart_it" \
    --use_cuda \
    --fp16 > bart_squad_it_finetune.log
