import re
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from .custom_datasets.qa_datasets import QA_SPECIAL_TOKENS


class GPTModel:

    def __init__(
            self,
            cache_dir,
            model_name="togethercomputer/GPT-JT-6B-v1",
            model_path=None
    ):
        if model_path is not None:
            # fineuned model checkpoint
            self.tokenizer = get_tokenizer(model_name, cache_dir)
            if "t5" in model_name:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            # public pretrained checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if "t5" in model_name:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def predict(
            self,
            input_text,
            max_length=1024,
            decoding_style="sampling",
            num_seqs=5,
            temperature=0.6,
            question_prefix="Question:",
            **kwargs
    ):
        with torch.no_grad():
            input_ids = torch.LongTensor(
                self.tokenizer.encode(input_text, verbose=False, max_length=max_length)
            ).unsqueeze(0).cuda()

            if decoding_style == "sampling":
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=temperature,
                    max_length=max_length,
                    num_return_sequences=num_seqs,
                    top_p=0.95)

            elif decoding_style == "beam_search":
                output_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    num_beams=5,
                    num_return_sequences=num_seqs,
                    early_stopping=True)

            outputs = []
            for output_id in output_ids:
                output_text = self.tokenizer.decode(output_id, skip_special_tokens=True)
                processed_output_text = post_processing(input_text, output_text, question_prefix)
                outputs.append(processed_output_text)
            return outputs


def get_tokenizer(model_name, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if "galactica" in model_name:
        tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "</s>"})
    elif "GPT-JT" in model_name:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token, "sep_token": "<|extratoken_100|>"})
    elif "codegen" in model_name:
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>", "sep_token": "<|endoftext|>"})
    elif "t5" in model_name:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    additional_special_tokens = (
        []
        if "additional_special_tokens" not in tokenizer.special_tokens_map
        else tokenizer.special_tokens_map["additional_special_tokens"]
    )
    additional_special_tokens = list(set(additional_special_tokens + list(QA_SPECIAL_TOKENS.values())))
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    return tokenizer


def post_processing(input_text, output_text, question_prefix):
    input_indices = [m.start() for m in re.finditer(question_prefix, input_text)]
    output_indices = [m.start() for m in re.finditer(question_prefix, output_text)]
    output_text = output_text[:output_indices[len(input_indices)]]
    return output_text
