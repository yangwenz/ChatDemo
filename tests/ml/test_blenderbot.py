import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test():
    model = AutoModelForCausalLM.from_pretrained("facebook/blenderbot-400M-distill")
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    print(model)
    print(tokenizer)


if __name__ == "__main__":
    test()
