import os
import time
import requests
from urllib.parse import urljoin
from dash import Input, Output, State, callback

URL = "http://{host}:{port}".format(
    host=os.getenv("CHATBOT_SERVER_HOST", "localhost"),
    port=os.getenv("CHATBOT_SERVER_PORT", 8081)
)
TIMEOUT = os.getenv("CHATBOT_SERVER_TIMEOUT", 20)
TASK_LIMIT = os.getenv("TASK_LIMIT", 50)


def query(payload):
    url = urljoin(URL, "queue_length")
    response = requests.get(url)
    if response.status_code != 200 or int(response.content) > TASK_LIMIT:
        return {"generated_text": "ERROR: chatbot server is too busy. Please try later."}

    url = urljoin(URL, "chat")
    response = requests.post(url, json=payload)
    task_id = response.json()["task_id"]
    for _ in range(TIMEOUT):
        result = requests.get(f"{url}/{task_id}")
        if result.status_code == 200:
            return result.json()
        time.sleep(1)
    return {"generated_text": "ERROR: Chatbot server timeouts."}


@callback(
    Output("user-input", "value"),
    [
        Input("submit", "n_clicks"),
        Input("user-input", "n_submit")
    ],
)
def clear_input(n_clicks, n_submit):
    return ""


def _process_chat_history(chat_history):
    chats = chat_history.split("<split>")
    past_user_inputs = [s[4:].strip() for s in chats if s.startswith("You:")]
    generated_responses = [s[4:].strip() for s in chats if s.startswith("Robot:")]
    return past_user_inputs, generated_responses


@callback(
    [
        Output("store-conversation", "data"),
        Output("loading-component", "children")
    ],
    [
        Input("submit", "n_clicks"),
        Input("user-input", "n_submit")
    ],
    [
        State("user-input", "value"),
        State("store-conversation", "data")
    ],
)
def run_chatbot(n_clicks, n_submit, user_input, chat_history):
    if n_clicks == 0 and n_submit is None:
        return "", None
    if user_input is None or user_input == "":
        return chat_history, None

    past_user_inputs, generated_responses = \
        _process_chat_history(chat_history)
    output = query({
        "inputs": {
            "past_user_inputs": past_user_inputs,
            "generated_responses": generated_responses,
            "text": user_input,
        }
    })
    model_output = output["generated_text"]
    chat_history += f"You: {user_input}<split>Robot: {model_output}<split>"
    return chat_history, None
