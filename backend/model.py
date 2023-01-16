
class BaseModel:

    def __init__(self, model_path=None):
        self.model_path = model_path

    def predict(self, inputs):
        return "Test test"


class TestModel(BaseModel):
    pass


class BlenderBotModel(BaseModel):

    def __init__(self, model_path=None):
        from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = "facebook/blenderbot-400M-distill"
        self.model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

    def predict(self, inputs):
        input_text = inputs["inputs"]["text"]
        input_ids = self.tokenizer([input_text], return_tensors="pt")
        reply_ids = self.model.generate(**input_ids)
        outputs = self.tokenizer.batch_decode(reply_ids)
        return outputs[0]


class ModelFactory:

    @staticmethod
    def create(model_cls="test"):
        if model_cls == "test":
            return TestModel
        elif model_cls in ["blender", "blenderbot"]:
            return BlenderBotModel
