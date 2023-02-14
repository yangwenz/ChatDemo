#!/bin/bash

MODEL_DIR=/export/share/wenzhuo/GPT-JT-6B-v1
OUTPUT_DIR=/export/share/wenzhuo/
NUM_GPU=1

python ./utils/t5/generate_triton_config.py --model_store ${OUTPUT_DIR} --hf_model_dir ${MODEL_DIR} --num_gpu ${NUM_GPU}
python ./utils/t5/huggingface_t5_ckpt_convert.py -o ${OUTPUT_DIR} -i ${MODEL_DIR} -i_g ${NUM_GPU}
