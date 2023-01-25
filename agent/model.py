import os
import torch
import logging
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


class BaseModel:

    def __init__(self, model_path=None):
        self.model_path = model_path

    def predict(self, inputs, **kwargs):
        return "Test test"


class TestModel(BaseModel):
    pass


class BlenderBotModel(BaseModel):

    def __init__(self, model_path=None):
        from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = "facebook/blenderbot-400M-distill"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.model.to(self.device)

    def predict(self, inputs, **kwargs):
        input_text = inputs["inputs"]["text"]
        input_ids = self.tokenizer([input_text], return_tensors="pt").to(self.device)
        reply_ids = self.model.generate(**input_ids)
        outputs = self.tokenizer.batch_decode(reply_ids)
        return outputs[0][4:-4]


class GPTJ(BaseModel):

    def __init__(self, model_path=None):
        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = "EleutherAI/gpt-j-6B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.model.to(self.device)

    def predict(self, inputs, **kwargs):
        input_text = inputs["inputs"]["text"]
        input_ids = torch.LongTensor(
            self.tokenizer.encode(input_text, verbose=False)
        ).unsqueeze(0).cuda()
        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100
        )
        return self.tokenizer.batch_decode(gen_tokens)[0]


class SearchModel(BaseModel):

    def __init__(self, model_path=None):
        from ml.model import GPTModel

        super().__init__(model_path)
        self.model = GPTModel(model_path)

    @staticmethod
    def get_model_input(
            inputs,
            question_prefix="Question:",
            answer_prefix="Answer:",
            sep="\n",
            prompt=None,
            **kwargs
    ):
        prompt = prompt + sep if prompt else ""
        inputs = inputs["inputs"]
        if "past_user_inputs" in inputs and "generated_responses" in inputs:
            for question, answer in zip(inputs["past_user_inputs"], inputs["generated_responses"]):
                if answer.startswith("ERROR:"):
                    continue
                prompt += f"{question_prefix} {question}{sep}{answer_prefix} {answer}{sep}"
        if "text" in inputs:
            prompt += f"{question_prefix} {inputs['text']}{sep}{answer_prefix} "
        return SearchModel._check_length(prompt, sep=sep)

    @staticmethod
    def _check_length(prompt, sep, max_length=1600):
        n = len(prompt.split())
        if n > max_length:
            sentences = prompt.split(sep)
            while len(sentences) > 4:
                question = sentences.pop(0)
                answer = sentences.pop(0)
                k = len(question.split()) + len(answer.split())
                n -= k
                if n < max_length * 0.95:
                    break
            return sep.join(sentences)
        return prompt

    def predict(self, inputs, prompt=None, **kwargs):
        input_text = self.get_model_input(inputs, prompt=prompt, **kwargs)
        outputs = self.model.predict(input_text, **kwargs)
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        else:
            return outputs


class TritonModel(BaseModel):

    def __init__(self, model_path=None):
        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = "togethercomputer/GPT-JT-6B-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.triton_url = "http://{host}:{port}".format(
            host=os.getenv("TRITON_HOST", "localhost"),
            port=os.getenv("TRITON_PORT", 8000)
        )
        self.triton_model_name = os.getenv("TRITON_MODEL_NAME", "fastertransformer")
        self.triton_model_version = os.getenv("TRITON_MODEL_VERSION", "1")
        logger.info(f"Triton url: {self.triton_url}")
        logger.info(f"Triton model name: {self.triton_model_name}")
        logger.info(f"Triton model version: {self.triton_model_version}")

        self.client = httpclient.InferenceServerClient(
            self.triton_url, concurrency=1, verbose=False)

    @staticmethod
    def _prepare_tensor(name, x):
        t = httpclient.InferInput(
            name, x.shape, np_to_triton_dtype(x.dtype))
        t.set_data_from_numpy(x)
        return t

    def predict(
            self,
            inputs,
            prompt=None,
            temperature=0.6,
            request_output_len=128,
            topk=1,
            topp=0,
            beam_width=1,
            len_penalty=1.0,
            repetition_penalty=1.0,
            question_prefix="Question:",
            answer_prefix="Answer:",
            **kwargs
    ):
        from ml.model import post_processing

        input_text = SearchModel.get_model_input(inputs, prompt=prompt, **kwargs)
        input_ids = np.expand_dims(
            self.tokenizer.encode(input_text, verbose=False), axis=0).astype(np.uint32)

        n = input_ids.shape[0]
        runtime_top_k = (topk * np.ones([n, 1])).astype(np.uint32)
        runtime_top_p = (topp * np.ones([n, 1])).astype(np.float32)
        temperature = (temperature * np.ones([n, 1])).astype(np.float32)
        len_penalty = (len_penalty * np.ones([n, 1])).astype(np.float32)
        repetition_penalty = (repetition_penalty * np.ones([n, 1])).astype(np.float32)
        beam_width = (beam_width * np.ones([n, 1])).astype(np.uint32)

        inputs = [
            self._prepare_tensor("input_ids", input_ids),
            self._prepare_tensor("input_lengths", np.array([[input_ids.shape[1]]], dtype=np.uint32)),
            self._prepare_tensor("request_output_len", np.array([[request_output_len]], dtype=np.uint32)),
            self._prepare_tensor("runtime_top_k", runtime_top_k),
            self._prepare_tensor("runtime_top_p", runtime_top_p),
            self._prepare_tensor("temperature", temperature),
            self._prepare_tensor("len_penalty", len_penalty),
            self._prepare_tensor("repetition_penalty", repetition_penalty),
            self._prepare_tensor("beam_width", beam_width),
        ]
        try:
            result = self.client.infer(
                model_name=self.triton_model_name,
                inputs=inputs,
                model_version=self.triton_model_version
            )
            output_ids = result.as_numpy("output_ids")[0][0]
            output_text = self.tokenizer.decode(output_ids)
            processed_output_text = post_processing(input_text, output_text, question_prefix)
            answer = processed_output_text.split(answer_prefix)[-1].strip()
            return answer

        except Exception as e:
            logger.warning(f"Triton inference error: {str(e)}")
            return "Inference server error. Please try again after some time."


class ModelFactory:

    @staticmethod
    def create(model_cls="test"):
        if model_cls == "test":
            return TestModel
        elif model_cls in ["blender", "blenderbot"]:
            return BlenderBotModel
        elif model_cls == "gptj":
            return GPTJ
        elif model_cls in ["search"]:
            return SearchModel
        elif model_cls == "triton":
            return TritonModel
        else:
            raise ValueError(f"Unknown model class: {model_cls}")
