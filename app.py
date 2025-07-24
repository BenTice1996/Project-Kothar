import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ctx
import re

# === Load Data from GitHub ===
CSV_URL = "https://dataverse.harvard.edu/api/access/datafile/11812470"
response = requests.get(CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
response.raise_for_status()
df['dateFiled'] = pd.to_datetime(df['dateFiled'], errors='coerce')
df['year'] = df['dateFiled'].dt.year
df['top_cluster'] = pd.to_numeric(df['top_cluster'], errors='coerce')

# === App Setup ===
app = dash.Dash(__name__)
server = app.server  # Required for deployment on services like Heroku/Render
app.title = "Copyright Infringement Topic Explorer"

# === Layout ===
app.layout = html.Div([
    html.H1("Topic-Modeled Copyright Infringement Cases Explorer"),

    html.Label("Search Opinion Text:"),
    dcc.Input(id="keyword-input", type="text", placeholder="Enter keyword in opinion text", style={"width": "100%"}),

    html.Label("Search Case Name:"),
    dcc.Input(id="case-name-input", type="text", placeholder="Enter part of case name", style={"width": "100%"}),

    html.Label("Filter by Topic:"),
    dcc.Dropdown(
        id="topic-dropdown",
        options=[{"label": f"Topic {i}: {df[df['top_cluster'] == i]['top_cluster_terms'].iloc[0]}", "value": i}
                 for i in sorted(df['top_cluster'].dropna().unique())],
        placeholder="Select a topic"
    ),

    html.Label("Filter by Year(s):"),
    dcc.Dropdown(
        id="year-dropdown",
        options=[{"label": str(y), "value": y} for y in sorted(df['year'].dropna().unique())],
        placeholder="Select year(s)",
        multi=True
    ),

    html.Hr(),

    html.Div(id="case-count-display", style={"fontWeight": "bold"}),

    html.Div(id="case-list"),

    html.Div([
        html.Button("Previous", id="prev-btn", n_clicks=0),
        html.Button("Next", id="next-btn", n_clicks=0),
        html.Span(id="page-number", style={"margin": "0 15px"})
    ], style={"marginTop": "20px"}),

    html.Br(),

    html.Button("Download CSV", id="download-btn"),
    dcc.Download(id="download-dataframe-csv"),

    html.Div(id="keyword-tally", style={"marginTop": "20px", "fontStyle": "italic"})
])

PAGE_SIZE = 5

@app.callback(
    [Output("case-count-display", "children"),
     Output("case-list", "children"),
     Output("page-number", "children"),
     Output("keyword-tally", "children"),
     Output("download-dataframe-csv", "data")],
    [Input("keyword-input", "value"),
     Input("case-name-input", "value"),
     Input("topic-dropdown", "value"),
     Input("year-dropdown", "value"),
     Input("prev-btn", "n_clicks"),
     Input("next-btn", "n_clicks"),
     Input("download-btn", "n_clicks")],
    [State("page-number", "children")]
)
def update_output(keyword, case_name_kw, topic, years, n_prev, n_next, n_download, page_state):
    filtered = df.copy()

    if years:
        filtered = filtered[filtered['year'].isin(years)]
    if topic is not None:
        filtered = filtered[filtered['top_cluster'] == topic]
    if keyword:
        keyword_lower = keyword.lower()
        filtered = filtered[filtered['opinion_text'].str.lower().str.contains(keyword_lower, na=False)]
    if case_name_kw:
        case_name_kw_lower = case_name_kw.lower()
        filtered = filtered[filtered['caseName'].str.lower().str.contains(case_name_kw_lower, na=False)]

    total_cases = len(filtered)
    count_text = f"{total_cases} case(s) match your filters."

    page = int(page_state) if page_state and page_state.isdigit() else 0
    page = max(0, min(page, total_cases // PAGE_SIZE))
    start, end = page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE
    paged = filtered.iloc[start:end]

    user_kw = keyword.lower() if keyword else None
    topic_kws = []
    if topic is not None:
        topic_terms = df[df['top_cluster'] == topic]['top_cluster_terms'].iloc[0]
        topic_kws = [kw.strip().lower() for kw in topic_terms.split(",")]

    case_divs = []
    for i, row in paged.iterrows():
        opinion = row['opinion_text']

        def highlight(text):
            if not text:
                return ""
            if user_kw:
                pattern_user = re.compile(rf'\b({re.escape(user_kw)})\b', flags=re.IGNORECASE)
                text = pattern_user.sub(r'<mark style="background-color: yellow">\1</mark>', text)
            for kw in topic_kws:
                pattern_topic = re.compile(rf'\b({re.escape(kw)})\b', flags=re.IGNORECASE)
                text = pattern_topic.sub(r'<mark style="background-color: lightblue">\1</mark>', text)
            return text

        opinion_html = highlight(opinion)

        case_divs.append(html.Div([
            html.H3(html.A(row['caseName'], href=row['absolute_url'], target="_blank")),
            html.Details([
                html.Summary("Show/Hide Opinion"),
                dcc.Markdown(opinion_html, style={"whiteSpace": "pre-wrap", "background": "#f9f9f9", "padding": "10px"}, dangerously_allow_html=True)
            ])
        ], style={"marginBottom": "30px"}))

    tally_block = ""
    if topic is not None and len(paged) == 1:
        opinion_text = paged.iloc[0]['opinion_text'].lower()
        counts = {kw: opinion_text.count(kw) for kw in topic_kws}
        tally_lines = [f"'{k}': {v} occurrence(s)" for k, v in counts.items()]
        tally_block = html.Pre("🧠 Topic keyword tally:\n" + "\n".join(tally_lines))

    if ctx.triggered_id == "download-btn":
        export_df = filtered[['caseName', 'absolute_url']]
        return count_text, case_divs, str(page), tally_block, dcc.send_data_frame(export_df.to_csv, "filtered_cases.csv", index=False)

    return count_text, case_divs, str(page), tally_block, dash.no_update

if __name__ == '__main__':
    print("🌐 Starting server at http://127.0.0.1:8050/")
    app.run_server(debug=True)
