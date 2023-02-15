import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def test():
    model_name = "/home/ywz/data/models/flan-t5-large"
    prompt = "A step by step recipe to make bolognese pasta:"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    start_time = time.time()
    input_ids = torch.LongTensor(tokenizer.encode(prompt, verbose=False)).unsqueeze(0).cuda()
    outputs = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=128,
        num_return_sequences=1
    )
    print(time.time() - start_time)
    for i, output_id in enumerate(outputs):
        output_text = tokenizer.decode(output_id, skip_special_tokens=True)
        print(f"----------------- Index {i} -----------------")
        print(output_text)


if __name__ == "__main__":
    test()
