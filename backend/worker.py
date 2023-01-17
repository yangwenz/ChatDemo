import os
import json
from celery import Celery
from backend.model import ModelFactory

model_cls = os.getenv("MODEL_CLASS", "blender")
model_path = os.getenv("MODEL_PATH", "")
model = ModelFactory.create(model_cls=model_cls)(model_path)

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")


@celery.task(name="generate_text")
def generate_text(inputs):
    try:
        outputs = model.predict(inputs)
    except Exception as e:
        outputs = f"ERROR: {str(e)}"
    return json.dumps({"generated_text": outputs})
