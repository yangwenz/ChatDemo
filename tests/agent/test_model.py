from agent.model import GPTJ


def test():
    past_user_inputs = ["How are you?", "Where are you from?"]
    generated_responses = ["Fine.", "United States."]
    inputs = {
        "inputs": {
            "past_user_inputs": past_user_inputs,
            "generated_responses": generated_responses,
            "text": "Do you watch NBA?",
        }
    }
    input_text = GPTJ._get_model_input(inputs)
    print(input_text)


if __name__ == "__main__":
    test()
