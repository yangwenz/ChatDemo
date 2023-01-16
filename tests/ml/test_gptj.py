import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test():
    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-1B", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-1B")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)


if __name__ == "__main__":
    test()
