import os
from agent.model import SearchModel
from agent.model import GPTJ


def test_1():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "../../agent/prompt_example.txt"), "r") as f:
        prompt = f.read()
    past_user_inputs = ["What is Streamforce?"]
    generated_responses = ["Streamforce is a live streaming service for IBM Watson, enabling presenters to broadcast their event to the world."]
    inputs = {
        "inputs": {
            "past_user_inputs": past_user_inputs,
            "generated_responses": generated_responses,
            "text": "How can I sign up as a presenter?",
        }
    }
    model = SearchModel("/home/ywz/data/models/GPT-JT-6B-v1")
    outputs = model.predict(inputs, prompt=prompt)
    print(outputs)


def test_2():
    model = GPTJ(model_path="/export/share/wenzhuo/gpt-j-6B")
    output = model.predict({
        "inputs": {"text": "Hello, how are you?"}
    })
    print(output)


if __name__ == "__main__":
    test_1()
