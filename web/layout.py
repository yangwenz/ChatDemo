import dash_bootstrap_components as dbc
from dash import html


def create_header(app, name):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("dash-logo.png"),
        style={"float": "right", "height": 60}
    )
    return dbc.Row([dbc.Col(title, md=8), dbc.Col(logo, md=4)])


def create_textbox(app, text, box="AI", name="Robot"):
    text = text.replace(f"{name}:", "").replace("You:", "")
    style = {
        "max-width": "60%",
        "width": "max-content",
        "padding": "5px 10px",
        "border-radius": 25,
        "margin-bottom": 20,
    }

    if box == "user":
        style["margin-left"] = "auto"
        style["margin-right"] = 0
        thumbnail = html.Img(
            src=app.get_asset_url("user.png"),
            style={
                "border-radius": 50,
                "height": 36,
                "margin-left": 5,
                "float": "right",
            },
        )
        textbox = dbc.Card(text, style=style, body=True, color="primary", inverse=True)
        return html.Div([thumbnail, textbox])

    elif box == "AI":
        style["margin-left"] = 0
        style["margin-right"] = "auto"
        thumbnail = html.Img(
            src=app.get_asset_url("robot.png"),
            style={
                "border-radius": 50,
                "height": 36,
                "margin-right": 5,
                "float": "left",
            },
        )
        textbox = dbc.Card(text, style=style, body=True, color="light", inverse=False)
        return html.Div([thumbnail, textbox])
    else:
        raise ValueError("Incorrect option for `box`.")


def create_conversation_box():
    return html.Div(
        html.Div(id="display-conversation"),
        style={
            "overflow-y": "auto",
            "display": "flex",
            "height": "calc(90vh - 132px)",
            "flex-direction": "column-reverse",
        },
    )


def create_controls():
    return dbc.InputGroup(
        children=[
            dbc.Input(id="user-input", placeholder="Write to the chatbot...", type="text"),
            dbc.Button("Submit", id="submit"),
        ]
    )
