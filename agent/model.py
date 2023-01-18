class BaseModel:

    def __init__(self, model_path=None):
        self.model_path = model_path

    def predict(self, inputs, **kwargs):
        return "Test test"


class TestModel(BaseModel):
    pass


class BlenderBotModel(BaseModel):

    def __init__(self, model_path=None):
        import torch
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
        from ml.model import GPTModel

        super().__init__(model_path)
        self.model = GPTModel(model_path)

    @staticmethod
    def _get_model_input(
            inputs,
            question_prefix="Question:",
            answer_prefix="Answer:",
            sep="\n",
            **kwargs
    ):
        prompt = ""
        inputs = inputs["inputs"]
        if "past_user_inputs" not in inputs or "generated_responses" not in inputs:
            return prompt
        for question, answer in zip(inputs["past_user_inputs"], inputs["generated_responses"]):
            if answer.startswith("ERROR:"):
                continue
            prompt += f"{question_prefix} {question}{sep}{answer_prefix} {answer}{sep}"
        if "text" in inputs:
            prompt += f"{question_prefix} {inputs['text']}{sep}{answer_prefix} "
        return prompt

    def predict(self, inputs, **kwargs):
        input_text = self._get_model_input(inputs, **kwargs)
        outputs = self.model.predict(input_text, **kwargs)
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        else:
            return outputs


class ModelFactory:

    @staticmethod
    def create(model_cls="test"):
        if model_cls == "test":
            return TestModel
        elif model_cls in ["blender", "blenderbot"]:
            return BlenderBotModel
        elif model_cls in ["gpt", "gptj", "gpt-j"]:
            return GPTJ
        else:
            raise ValueError(f"Unknown model class: {model_cls}")
