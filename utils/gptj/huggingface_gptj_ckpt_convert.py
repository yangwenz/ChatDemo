import os
from argparse import ArgumentParser
from os import makedirs
import numpy as np
from pathlib import Path

import torch
import configparser
from transformers import PretrainedConfig

torch.set_printoptions(linewidth=130, sci_mode=False)
np.set_printoptions(linewidth=130, suppress=True)


# https://github.com/NVIDIA/FasterTransformer/blob/main/examples/pytorch/gptj/utils/huggingface_gptj_ckpt_convert.py
# This converter is used to convert the huggingface gpt-j-6B model
# in https://huggingface.co/EleutherAI/gpt-j-6B/blob/main/pytorch_model.bin.

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def savebin(param, save_path, weight_data_type):
    if isinstance(param, torch.Tensor):
        param = param.cpu().float().numpy()
    data_type = get_weight_data_type(weight_data_type)
    np.squeeze(param).astype(data_type).tofile(save_path + ".bin")


def param2file(pt_param, layer_id, save_dir, dest_key, weight_data_type):
    base_n = save_dir + "/model.layers." + str(layer_id) + "."
    save_path = base_n + dest_key
    savebin(pt_param, save_path, weight_data_type)


def param2distributed(
        pt_param,
        layer_id,
        save_dir,
        dest_key,
        n_inference_gpus,
        split_axis,
        weight_data_type
):
    np_param = pt_param.cpu().float().numpy()
    base_n = save_dir + "/model.layers." + str(layer_id) + "."
    save_path = base_n + dest_key
    split_param = np.split(np_param, n_inference_gpus, axis=split_axis)
    for i, p in enumerate(split_param):
        savebin(p, save_path + f".{i}", weight_data_type)


def save(w, save_dir, n_inference_gpus, n_layers, layer_id, weight_data_type):
    makedirs(save_dir, exist_ok=True)

    savebin(w['transformer.wte.weight'], save_dir + "/model.wte", weight_data_type)
    l = layer_id
    print(f"Saving layer {l + 1} / {n_layers}")
    base_k = "transformer.h." + str(l) + "."
    param2file(
        w[base_k + "ln_1.bias"],
        l, save_dir, "input_layernorm.bias",
        weight_data_type
    )
    param2file(
        w[base_k + "ln_1.weight"],
        l, save_dir, "input_layernorm.weight",
        weight_data_type
    )
    param2distributed(
        w[base_k + "mlp.fc_in.weight"].T,
        l, save_dir, "mlp.dense_h_to_4h.weight",
        n_inference_gpus, split_axis=-1,  # split fast indx
        weight_data_type=weight_data_type
    )
    param2distributed(
        w[base_k + "mlp.fc_in.bias"],
        l, save_dir, "mlp.dense_h_to_4h.bias",
        n_inference_gpus, split_axis=-1,  # split fast indx
        weight_data_type=weight_data_type
    )
    param2distributed(
        w[base_k + "mlp.fc_out.weight"].T,
        l, save_dir, "mlp.dense_4h_to_h.weight",
        n_inference_gpus, split_axis=0,  # split slow indx
        weight_data_type=weight_data_type
    )
    param2file(
        w[base_k + "mlp.fc_out.bias"],
        l, save_dir, "mlp.dense_4h_to_h.bias",
        weight_data_type
    )
    param2distributed(
        w[base_k + "attn.out_proj.weight"].T,
        l, save_dir, "attention.dense.weight",
        n_inference_gpus, split_axis=0,  # split slow indx
        weight_data_type=weight_data_type
    )
    QKV_w = torch.stack([
        w[base_k + "attn.q_proj.weight"],
        w[base_k + "attn.k_proj.weight"],
        w[base_k + "attn.v_proj.weight"],
    ])  # [qkv, n_heads * dim_head, latent_space]
    QKV_w = QKV_w.permute(2, 0, 1)
    param2distributed(
        QKV_w, l, save_dir, "attention.query_key_value.weight",
        n_inference_gpus, split_axis=-1,  # split fast indx
        weight_data_type=weight_data_type
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert GPT-J slim checkpoint to FasterTransformer",
    )
    parser.add_argument(
        "--output-dir", help="Folder where binary files are stored", default="gpt-j-6B/c-models/"
    )
    parser.add_argument(
        "--ckpt-dir", help="File of GPT-J huggingface checkpoint", default="gpt-j-6B/"
    )
    parser.add_argument(
        "--n-inference-gpus", help="Number of GPUs used for inference runtime", default=1, type=int
    )
    parser.add_argument(
        "--n-layers", help="Number of GPT-J decoder layer", default=28, type=int
    )
    parser.add_argument(
        "--weight_data_type", type=str, default="fp16", choices=["fp32", "fp16"]
    )
    parser.add_argument(
        "--version", "-v", type=int, help="Model version", default=1
    )
    args = parser.parse_args()

    args.ckpt_dir = args.ckpt_dir.rstrip("/")
    model_name = os.path.basename(args.ckpt_dir)
    out_path = os.path.join(args.output_dir, f"{model_name}-{args.n_inference_gpus}gpu", "fastertransformer")
    output_dir = os.path.join(out_path, f"{args.version}")
    print(f"saving to {output_dir}")

    config_file = os.path.join(args.ckpt_dir, "config.json")
    hf_config = PretrainedConfig.from_json_file(config_file).to_dict()

    # NOTE: save parameters to config files (loaded by triton backends)
    config = configparser.ConfigParser()
    config["gptj"] = {}
    try:
        config["gptj"]["model_name"] = "gptj" if hf_config["_name_or_path"] == '' else hf_config["_name_or_path"]
        config["gptj"]["head_num"] = str(hf_config["n_head"])
        n_embd = hf_config["n_embd"]
        config["gptj"]["size_per_head"] = str(n_embd // hf_config["n_head"])
        config["gptj"]["inter_size"] = str(n_embd * 4)
        config["gptj"]["num_layer"] = str(hf_config["n_layer"])
        rotary_dim = n_embd // hf_config["n_head"] if hf_config["rotary_dim"] is None else hf_config["rotary_dim"]
        config["gptj"]["rotary_embedding"] = str(hf_config["rotary_dim"])
        config["gptj"]["vocab_size"] = str(hf_config["vocab_size"])
        config["gptj"]["start_id"] = str(hf_config["bos_token_id"])
        config["gptj"]["end_id"] = str(hf_config["eos_token_id"])
        config["gptj"]["weight_data_type"] = args.weight_data_type
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        with open(output_dir + "/config.ini", 'w') as configfile:
            config.write(configfile)
    except Exception as e:
        print(f"Fail to save the config in config.ini.")
        raise e

    ckpt_file = os.path.join(args.ckpt_dir, "pytorch_model.bin")
    checkpoint = torch.load(ckpt_file)
    print(f"loading from {ckpt_file}")

    for layer_idx in range(args.n_layers):
        save(checkpoint, output_dir, args.n_inference_gpus, args.n_layers, layer_idx, args.weight_data_type)
    savebin(checkpoint['transformer.ln_f.weight'], output_dir + "/model.final_layernorm.weight", args.weight_data_type)
    savebin(checkpoint['transformer.ln_f.bias'], output_dir + "/model.final_layernorm.bias", args.weight_data_type)
    savebin(checkpoint['lm_head.weight'], output_dir + "/model.lm_head.weight", args.weight_data_type)
    savebin(checkpoint['lm_head.bias'], output_dir + "/model.lm_head.bias", args.weight_data_type)
    print("done")
