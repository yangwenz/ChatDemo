import json
import argparse
from flask import Flask, request, abort
from flask_compress import Compress
from backend.worker import generate_text

app = Flask(__name__)
Compress(app)


@app.route("/", methods=["GET"])
def ping():
    return "OK", 200


@app.route("/chat", methods=["POST"])
def chat():
    if not request.json:
        abort(400)
    task = generate_text.delay(request.json)
    print(task)
    return json.dumps({"task_id": task.id}), 200


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Host")
    parser.add_argument("--port", default=8081, type=int, help="Port")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
