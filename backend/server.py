import argparse
from flask import Flask, request, abort
from .model import Model


model = Model()
app = Flask(__name__)


@app.route("/", methods=["GET"])
def ping():
    return "OK", 200


@app.route("/chat", methods=["POST"])
def chat():
    if not request.json:
        abort(400)
    try:
        outputs = model.predict(request.json)
    except Exception:
        outputs = "ERROR"
    return outputs, 200


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Host")
    parser.add_argument("--port", default=8081, type=int, help="Port")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
