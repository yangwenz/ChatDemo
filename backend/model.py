
class BaseModel:

    def __init__(self, model_path=None):
        self.model_path = model_path

    def predict(self, inputs):
        return "Test test"


class TestModel(BaseModel):
    pass


class BlenderBotModel(BaseModel):
    pass


class ModelFactory:

    @staticmethod
    def create(model_cls="test"):
        if model_cls == "test":
            return TestModel
        elif model_cls in ["blender", "blenderbot"]:
            return BlenderBotModel
