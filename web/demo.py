import json
import os
import time
import requests
import streamlit as st
from urllib.parse import urljoin
from web.chat import message

st.set_page_config(
    page_title="Streamlit Chat - Demo",
    page_icon=":robot:"
)
st.header("GPTJ Chatbot - Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

URL = "http://{host}:{port}".format(
    host=os.getenv("CHATBOT_SERVER_HOST", "localhost"),
    port=os.getenv("CHATBOT_SERVER_PORT", 8081)
)
TIMEOUT = os.getenv("CHATBOT_SERVER_TIMEOUT", 20)


def query(payload):
    url = urljoin(URL, "chat")
    response = requests.post(url, json=payload)
    task_id = response.json()["task_id"]
    for _ in range(TIMEOUT):
        result = requests.get(f"{url}/{task_id}")
        if result.status_code == 200:
            return result.json()
        time.sleep(1)
    return {"generated_text": "Chatbot server timeouts."}


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = query({
        "inputs": {
            "past_user_inputs": st.session_state.past,
            "generated_responses": st.session_state.generated,
            "text": user_input,
        }, "parameters": {"repetition_penalty": 1.33},
    })
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output["generated_text"])

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
