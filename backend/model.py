
class BaseModel:

    def __init__(self, model_path=None):
        self.model_path = model_path

    def predict(self, inputs):
        return "Test test"


class TestModel:
    pass


class ModelFactory:

    @staticmethod
    def create(model_cls="test"):
        if model_cls == "test":
            return TestModel
