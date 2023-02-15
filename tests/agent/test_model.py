import os
from agent.model import SearchModel
from agent.model import GPTJ
from agent.model import TritonGPTJModel, TritonT5Model


def test_1():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "../../agent/prompt_example.txt"), "r") as f:
        prompt = f.read()
    past_user_inputs = []
    generated_responses = []
    inputs = {
        "inputs": {
            "past_user_inputs": past_user_inputs,
            "generated_responses": generated_responses,
            "text": "How can I sign up as a presenter?",
        }
    }
    model = SearchModel("/export/share/wenzhuo/GPT-JT-6B-v1")
    outputs = model.predict(inputs, prompt=prompt)
    print(outputs)


def test_2():
    model = GPTJ(model_path="/export/share/wenzhuo/gpt-j-6B")
    output = model.predict({
        "inputs": {"text": "Hello, how are you?"}
    })
    print(output)


def test_3():
    model = TritonGPTJModel()
    output = model.predict({
        "inputs": {"text": "Hello, how are you?"}
    })
    print(output)


def test_4():
    model = TritonT5Model()
    output = model.predict({
        "inputs": {"text": "A step by step recipe to make bolognese pasta:"}
    })
    print(output)


if __name__ == "__main__":
    test_4()
