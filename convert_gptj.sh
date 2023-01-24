#!/bin/bash

MODEL_DIR=/home/ywz/data/models/GPT-JT-6B-v1
OUTPUT_DIR=/home/ywz/data/models/ft
NUM_GPU=1

python ./utils/gptj/generate_triton_config.py --model_store ${OUTPUT_DIR} --hf_model_dir ${MODEL_DIR} --num_gpu ${NUM_GPU}
python ./utils/gptj/huggingface_gptj_ckpt_convert.py --output-dir ${OUTPUT_DIR} --ckpt-dir ${MODEL_DIR} --n-inference-gpus ${NUM_GPU}
