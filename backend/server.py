import os
import json
import logging
import argparse
from flask import Flask, request, abort
from flask_compress import Compress
from agent.worker import app as celery_app
from agent.worker import generate_text

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)
TIMEOUT = os.getenv("CHATBOT_SERVER_TIMEOUT", 20)

app = Flask(__name__)
Compress(app)


@app.route("/", methods=["GET"])
def ping():
    return "OK", 200


@app.route("/chat", methods=["POST"])
def chat():
    try:
        task = generate_text.delay(request.json)
        result = celery_app.AsyncResult(task.id)
        result = json.loads(result.get(timeout=TIMEOUT))
        response = {"task_id": task.id, "status": "Success"}
        response.update(result)
        return response, 200
    except Exception as e:
        logger.warning(f"ERROR: {str(e)}")
        abort(400)


@app.route("/chat/<task_id>", methods=["GET"])
def get_status(task_id):
    try:
        task = celery_app.AsyncResult(task_id)
        if not task.ready():
            response = {"task_id": str(task_id), "status": "Processing"}
            status_code = 202
        else:
            result = json.loads(task.get())
            response = {"task_id": str(task_id), "status": "Success"}
            response.update(result)
            status_code = 200
        return json.dumps(response), status_code
    except Exception as e:
        logger.warning(f"ERROR (task_id = {task_id}): {str(e)}")
        abort(400)


@app.route("/queue_length", methods=["GET"])
def get_task_queue_length():
    try:
        with celery_app.pool.acquire(block=True) as conn:
            queue_name = celery_app.conf.task_default_queue
            return str(conn.default_channel.client.llen(queue_name)), 200
    except Exception as e:
        logger.warning(f"ERROR in get_task_queue: {str(e)}")
        abort(400)


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Host")
    parser.add_argument("--port", default=8081, type=int, help="Port")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
