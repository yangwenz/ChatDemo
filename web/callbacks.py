from dash import Input, Output, State, callback


@callback(
    Output("user-input", "value"),
    [
        Input("submit", "n_clicks"),
        Input("user-input", "n_submit")
    ],
)
def clear_input(n_clicks, n_submit):
    return ""


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

    name = "Robot"
    # First add the user input to the chat history
    chat_history += f"You: {user_input}<split>{name}:"
    '''
    response = openai.Completion.create(
        engine="davinci",
        prompt=model_input,
        max_tokens=250,
        stop=["You:"],
        temperature=0.9,
    )
    model_output = response.choices[0].text.strip()
    '''
    model_output = "TEST TEST"

    chat_history += f"{model_output}<split>"
    return chat_history, None
