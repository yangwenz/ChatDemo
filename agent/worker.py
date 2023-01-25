import os
import json
import logging
from celery import Celery, Task
from celery.signals import beat_init, worker_ready, worker_shutdown
from agent.probe import LivenessProbe, READINESS_FILE

redis_url = "redis://{host}:{port}".format(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=os.getenv("REDIS_PORT", 6379)
)

app = Celery(__name__)
app.conf.broker_url = redis_url
app.conf.result_backend = redis_url
app.steps["worker"].add(LivenessProbe)

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)
logger.info(f"Broker URL: {app.conf.broker_url}")
logger.info(f"Backend URL: {app.conf.result_backend}")


class MLTask(Task):
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
            from agent.model import ModelFactory
            model_cls = os.getenv("MODEL_CLASS", "blender")
            model_path = os.getenv("MODEL_PATH", "")
            logger.info(f"Loading model {model_cls}")
            self.model = ModelFactory.create(model_cls=model_cls)(model_path)
            logger.info("Model loaded")

            logger.info(f"Loading prompt examples")
            folder = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(folder, "prompt_example.txt"), "r") as f:
                self.prompt = f.read()
            logger.info("Prompt loaded")

        return self.run(*args, **kwargs)


@app.task(
    ignore_result=False,
    bind=True,
    base=MLTask
)
def generate_text(self, inputs):
    try:
        outputs = self.model.predict(inputs, prompt=self.prompt)
    except Exception as e:
        outputs = f"ERROR: {str(e)}"
    return json.dumps({"generated_text": outputs})


@worker_ready.connect
def worker_ready(**kwargs):
    READINESS_FILE.touch()


@worker_shutdown.connect
def worker_shutdown(**kwargs):
    READINESS_FILE.unlink(missing_ok=True)
