import os
from agent.model import SearchModel
from agent.model import GPTJ


def test_1():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "../../agent/prompt_example.txt"), "r") as f:
        prompt = f.read()
    past_user_inputs = ["How are you?", "Where are you from?"]
    generated_responses = ["Fine.", "United States."]
    inputs = {
        "inputs": {
            "past_user_inputs": past_user_inputs,
            "generated_responses": generated_responses,
            "text": "Do you know Salesforce?",
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
