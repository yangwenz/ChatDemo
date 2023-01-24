import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test():
    model_name = "/home/ywz/data/models/GPT-JT-6B-v1"
    prompt = "Question: Hello, how are you? Answer:"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    start_time = time.time()
    input_ids = torch.LongTensor(tokenizer.encode(prompt, verbose=False)).unsqueeze(0).cuda()
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
        num_return_sequences=1
    )
    print(time.time() - start_time)
    for i, output_id in enumerate(gen_tokens):
        output_text = tokenizer.decode(output_id, skip_special_tokens=True)
        print(f"----------------- Index {i} -----------------")
        print(output_text)


if __name__ == "__main__":
    test()
