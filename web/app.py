import os
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output
from web.layout import create_banner, create_textbox, \
    create_conversation_box, create_controls

import web.callbacks

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Chatbot",
)
app.config["suppress_callback_exceptions"] = True
server = app.server

app.layout = dbc.Container(
    fluid=False,
    children=[
        create_banner(app),
        html.Br(),
        create_conversation_box(),
        create_controls(),
        html.Br(),
        dbc.Spinner(html.Div(id="loading-component")),
        dcc.Store(id="store-conversation", data=""),
    ],
)


@app.callback(
    Output("display-conversation", "children"),
    Input("store-conversation", "data")
)
def update_display(chat_history):
    return [
        create_textbox(app, text, box="user") if i % 2 == 0 else create_textbox(app, text, box="AI")
        for i, text in enumerate(chat_history.split("<split>")[:-1])
    ]


if __name__ == "__main__":
    host = os.getenv("HOST", "localhost")
    port = os.getenv("PORT", "8080")
    app.run_server(host=host, port=port, debug=False)
