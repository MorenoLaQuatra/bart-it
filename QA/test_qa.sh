# squad_it
python test_qa.py \
    --model_path "bart_squad_it/best_model/" \
    --tokenizer_path "bart_squad_it/tokenizer/" \
    --dataset_name "squad_it" \
    --use_cuda \
    --print_samples > bart_squad_it_test.log
