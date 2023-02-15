import time
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer
from tritonclient.utils import np_to_triton_dtype


def prepare_tensor(name, input):
    t = httpclient.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


# Initizlize client
model_name = "togethercomputer/GPT-JT-6B-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
client = httpclient.InferenceServerClient("localhost:8000", concurrency=1, verbose=False)

model_metadata = client.get_model_metadata(model_name="fastertransformer", model_version="1")
model_config = client.get_model_config(model_name="fastertransformer", model_version="1")
print(model_metadata)
print(model_config)

# Prepare tokens for sending to the server
input_text = "Question: Hello, how are you? Answer:"
input_ids = np.expand_dims(tokenizer.encode(input_text, verbose=False), axis=0).astype(np.uint32)
print(input_ids)

topk = 1
topp = 0
beam_width = 1
return_log_probs = False

n = input_ids.shape[0]
runtime_top_k = (topk * np.ones([n, 1])).astype(np.uint32)
runtime_top_p = (topp * np.ones([n, 1])).astype(np.float32)
temperature = np.ones([n, 1]).astype(np.float32)
len_penalty = np.ones([n, 1]).astype(np.float32)
repetition_penalty = np.ones([n, 1]).astype(np.float32)
beam_width = (beam_width * np.ones([n, 1])).astype(np.uint32)


inputs = [
    prepare_tensor("input_ids", input_ids),
    prepare_tensor("input_lengths", np.array([[input_ids.shape[1]]], dtype=np.uint32)),
    prepare_tensor("request_output_len", np.array([[64]], dtype=np.uint32)),
    prepare_tensor("runtime_top_k", runtime_top_k),
    prepare_tensor("runtime_top_p", runtime_top_p),
    prepare_tensor("temperature", temperature),
    prepare_tensor("len_penalty", len_penalty),
    prepare_tensor("repetition_penalty", repetition_penalty),
    prepare_tensor("beam_width", beam_width),
]

# Sending request
start_time = time.time()
result = client.infer("fastertransformer", inputs, model_version="1")
output0 = result.as_numpy("output_ids")[0]
output1 = result.as_numpy("sequence_length")[0]
print(f"Inference time: {time.time() - start_time}")

print("============After fastertransformer============")
print(output0)
print(output1)

for i, output_id in enumerate(output0):
    output_text = tokenizer.decode(output_id, skip_special_tokens=True)
    print(f"----------------- Index {i} -----------------")
    print(output_text)
