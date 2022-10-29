# --------------------------------------------------------------------
# This script is used to evaluate competitors on different datasets.
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