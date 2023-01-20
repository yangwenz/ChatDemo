import os
from agent.model import SearchModel


def test():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "../../agent/prompt_example.txt"), "r") as f:
        prompt = f.read()
    output = SearchModel._check_length(prompt, "\n", max_length=1000)
    print(output)


if __name__ == "__main__":
    test()
