# --------------------------------------------------------------------
# This script is used to fine-tune the model on different datasets.
# --------------------------------------------------------------------
:'
# ARTeLab/ilpost 
python evaluate_summarization.py \
    --model_path "ilpost_checkpoints/best_model" \
    --tokenizer_path "../tokenizer_bart_it" \
    --dataset_name "ARTeLab/ilpost" \
    --max_length 128 \
    --use_cuda > ilpost_evaluate_test.log

# ARTeLab/fanpage
python evaluate_summarization.py \
    --model_path "fanpage_checkpoints/best_model" \
    --tokenizer_path "../tokenizer_bart_it" \
    --dataset_name "ARTeLab/fanpage" \
    --max_length 128 \
    --use_cuda > fanpage_evaluate_test.log

# ARTeLab/mlsum-it
python evaluate_summarization.py \
    --model_path "mlsum_checkpoints/best_model" \
    --tokenizer_path "../tokenizer_bart_it" \
    --dataset_name "ARTeLab/mlsum-it" \
    --max_length 128 \
    --use_cuda > mlsum_evaluate_test.log

'

# Silvia/WITS
python evaluate_summarization.py \
    --model_path "wits_checkpoints/best_model" \
    --tokenizer_path "../tokenizer_bart_it" \
    --dataset_name "Silvia/WITS" \
    --max_length 128 \
    --use_cuda > wits_evaluate_test.log