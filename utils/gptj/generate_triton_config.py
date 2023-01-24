import os
import argparse
from string import Template
from transformers import GPTJConfig

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "config_template.pbtxt")

parser = argparse.ArgumentParser("Create Triton config files for GPT-J models")
parser.add_argument("--template", default=CONFIG_TEMPLATE_PATH, help="Path to the config template")
parser.add_argument("--model_store", required=True, help="Path to the Triton model store")
parser.add_argument("--hf_model_dir", required=True, help="Path to HF model directory")
parser.add_argument("-n", "--num_gpu", help="Number of GPUs to use", type=int, default=1)
args = parser.parse_args()

args.hf_model_dir = args.hf_model_dir.rstrip("/")
config = GPTJConfig.from_pretrained(args.hf_model_dir)
with open(args.template, 'r') as f:
    template = Template(f.read())

model_name = os.path.basename(args.hf_model_dir)
params = {
    "tensor_para_size": args.num_gpu,
    "name": model_name
}
model_dir = os.path.join(args.model_store, f"{model_name}-{args.num_gpu}gpu")
weights_path = os.path.join(model_dir, f"{args.num_gpu}-gpu")
params["checkpoint_path"] = weights_path
triton_config = template.substitute(params)
assert "${" not in triton_config

os.makedirs(weights_path, exist_ok=True)
config_path = os.path.join(model_dir, "config.pbtxt")
with open(config_path, "w") as f:
    f.write(triton_config)

print('==========================================================')
print(f'Created config file for {model_name}')
print(f'  Config:      {config_path}')
print(f'  Weights:     {weights_path}')
print(f'  Store:       {args.model_store}')
print(f'  Num GPU:     {args.num_gpu}')
print('==========================================================')
