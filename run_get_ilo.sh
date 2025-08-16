export CUDA_VISIBLE_DEVICES=4
python get_ilo.py \
        --model_name meta-llama/Meta-Llama-3.1-8B \
        --neighbor_k 5 \
        --neighbor_threshold 3 \
        --metric cosine