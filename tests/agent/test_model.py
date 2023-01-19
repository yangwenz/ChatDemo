from agent.model import SearchModel
from agent.model import GPTJ


def test_1():
    past_user_inputs = ["How are you?", "Where are you from?"]
    generated_responses = ["Fine.", "United States."]
    inputs = {
        "inputs": {
            "past_user_inputs": past_user_inputs,
            "generated_responses": generated_responses,
            "text": "Do you watch NBA?",
        }
    }
    input_text = SearchModel._get_model_input(inputs)
    print(input_text)


def test_2():
    model = GPTJ(model_path="/export/share/wenzhuo/gpt-j-6B")
    output = model.predict({
        "inputs": {"text": "Hello, how are you?"}
    })
    print(output)


if __name__ == "__main__":
    test_2()
