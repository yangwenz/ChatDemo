import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration


def test():
    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    input_text = "My friends are cool but they eat too many carbs."
    inputs = tokenizer([input_text], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    print(tokenizer.batch_decode(reply_ids))


if __name__ == "__main__":
    test()
