import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTModel:

    def __init__(self, model_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def predict(
            self,
            input_text,
            max_length=2048,
            decoding_style="sampling",
            num_seqs=1,
            temperature=0.6,
            question_prefix="Question:",
            answer_prefix="Answer:",
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
                output_text = self.tokenizer.decode(output_id)
                processed_output_text = post_processing(input_text, output_text, question_prefix)
                answer = processed_output_text.split(answer_prefix)[-1].strip()
                outputs.append(answer)
            return outputs


def post_processing(input_text, output_text, question_prefix):
    input_indices = [m.start() for m in re.finditer(question_prefix, input_text)]
    output_indices = [m.start() for m in re.finditer(question_prefix, output_text)]
    output_text = output_text[:output_indices[len(input_indices)]]
    return output_text
