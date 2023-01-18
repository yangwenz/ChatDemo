import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test():
    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = torch.LongTensor(tokenizer.encode(prompt, verbose=False)).unsqueeze(0).cuda()
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
        num_return_sequences=5
    )
    # gen_text = tokenizer.batch_decode(gen_tokens)[0]
    for i, output_id in enumerate(gen_tokens):
        output_text = tokenizer.decode(output_id, skip_special_tokens=True)
        print(f"----------------- Index {i} -----------------")
        print(output_text)


if __name__ == "__main__":
    test()
