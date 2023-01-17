import os
import json
import logging
from celery import Celery, Task
from backend.model import ModelFactory

model_cls = os.getenv("MODEL_CLASS", "blender")
model_path = os.getenv("MODEL_PATH", "")
broker_url = os.environ.get("BROKER_URL", "redis://localhost:6379")
backend_url = os.environ.get("BACKEND_URL", "redis://localhost:6379")

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

app = Celery(__name__, backend=backend_url, broker=broker_url)


class GenerationTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """
    abstract = True

    def __init__(self):
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.model:
            logger.info(f"Loading model {model_cls}")
            self.model = ModelFactory.create(model_cls=model_cls)(model_path)
            logger.info("Model loaded")
        return self.run(*args, **kwargs)


@app.task(
    ignore_result=False,
    bind=True,
    base=GenerationTask
)
def generate_text(self, inputs):
    try:
        outputs = self.model.predict(inputs)
    except Exception as e:
        outputs = f"ERROR: {str(e)}"
    return json.dumps({"generated_text": outputs})
