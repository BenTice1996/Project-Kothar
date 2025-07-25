import os
import threading
import time
import requests
import pandas as pd
import dash
from dash import dcc, html, Input, Output

# === Lazy load CSV from Dataverse ===
CSV_URL = "https://dataverse.harvard.edu/api/access/datafile/11816137"

def load_data():
    response = requests.get(CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.content.decode("utf-8")), usecols=["caseName", "absolute_url", "opinion_text", "top_cluster", "top_cluster_terms", "dateFiled"])

df = load_data()
df["dateFiled"] = pd.to_datetime(df["dateFiled"], errors="coerce")
df["year"] = df["dateFiled"].dt.year

# === Dash App ===
app = dash.Dash(__name__)
server = app.server  # Required for HF

app.layout = html.Div([
    html.H1("Legal Topic Explorer"),
    dcc.Input(id="keyword", type="text", placeholder="Enter keyword"),
    html.Div(id="output")
])

@app.callback(
    Output("output", "children"),
    Input("keyword", "value")
)
def update_output(kw):
    if not kw:
        return "Enter a keyword to search."
    matches = df[df["opinion_text"].str.contains(kw, case=False, na=False)]
    return [html.Div([
        html.A(row["caseName"], href=row["absolute_url"], target="_blank")
    ]) for _, row in matches.head(5).iterrows()]

# === Run server on port 7860 ===
def run():
    app.run_server(host="0.0.0.0", port=7860)

# === Required Gradio interface ===
def launch_dash():
    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(3)

launch_dash()
