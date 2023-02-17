#!/bin/bash

MODEL_DIR=/home/ywz/data/models/flan-t5-large
OUTPUT_DIR=/home/ywz/data/models/
NUM_GPU=1

python ./utils/t5/generate_triton_config.py --model_store ${OUTPUT_DIR} --hf_model_dir ${MODEL_DIR} --num_gpu ${NUM_GPU}
python ./utils/t5/huggingface_t5_ckpt_convert.py -o ${OUTPUT_DIR} -i ${MODEL_DIR} -i_g ${NUM_GPU} -weight_data_type fp16
