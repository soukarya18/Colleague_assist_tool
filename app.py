from __future__ import annotations
import dash
import json
import re 
from datetime import datetime
from pathlib import Path
from typing import Any

from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from metric_manager import load_metrics, save_metrics





APP_NAME = "Colleague Assist"
RESULTS_DIR = Path("results")
TRANSCRIPTS_DIR = Path("transcripts")


def load_results() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if not RESULTS_DIR.exists():
        return results

    for result_file in RESULTS_DIR.glob("*_analysis.json"):
        try:
            with result_file.open("r", encoding="utf-8") as file:
                data = json.load(file)

            recording = data.get(
                "recording",
                result_file.stem.replace("_analysis", ""),
            )
            transcript_file = (
                TRANSCRIPTS_DIR /
                f"{recording}.txt"
            )

            duration_seconds = (
                get_transcript_duration(
                    transcript_file
                )
            )
            results.append(
                {
                    "recording": recording,
                    "score": float(data.get("total_score", 0) or 0),
                    "maximum_score": float(data.get("maximum_score", 0) or 0),
                    "metrics": data.get("metrics", []),
                    "date": datetime.fromtimestamp(result_file.stat().st_mtime),
                    "duration_seconds": duration_seconds,
                    "duration" :format_duration(
                        duration_seconds
                    ) , 
                    
                    "call_type": "Customer Support",
                    "result_file": result_file,
                    "transcript_file": TRANSCRIPTS_DIR / f"{recording}.txt",
                }
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue

    results.sort(key=lambda item: item["date"], reverse=True)
    return results


def get_transcript_duration(
    transcript_path: Path
) -> float:

    if not transcript_path.exists():
        return 0.0

    try:
        lines = transcript_path.read_text(
            encoding="utf-8"
        ).splitlines()

        last_end_time = 0.0

        for line in lines:

            match = re.search(
                r"--> ([0-9.]+)s",
                line
            )

            if match:

                end_time = float(
                    match.group(1)
                )

                last_end_time = max(
                    last_end_time,
                    end_time
                )

        return last_end_time

    except OSError:
        return 0.0


def format_duration(
    seconds: float
) -> str:

    if seconds <= 0:
        return "—"

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    hours = minutes // 60
    minutes = minutes % 60

    if hours > 0:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )




def score_percent(item: dict[str, Any]) -> float:
    maximum = item.get("maximum_score", 0)
    score =float(item.get("score" , 0) or 0)
    if maximum<=0: 
        return 0.0
    if 0<=score <=100 and score>maximum:
        return score 

    return (score/maximum )*100

    # return item["score"] / maximum * 100 if maximum else 0.0




def score_bucket(score: float) -> str:
    if score >= 80:
        return "Green"
    if score >= 60:
        return "Amber"
    return "Red"


def stat_card(title: str, value: str, subtitle: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(title, className="stat-title"),
            html.Div(value, className="stat-value"),
            html.Div(subtitle, className="stat-subtitle"),
        ],
        className="stat-card",
    )


def page_shell(content: Any) -> html.Div:
    return html.Div(content, className="page-content")


def dashboard_page() -> Any:
    results = load_results()
    total_calls = len(results)
    avg_score = (
        sum(score_percent(item) for item in results) / total_calls
        if total_calls
        else 0
    )
    quality_rate = (
        sum(score_percent(item) >= 85 for item in results) / total_calls * 100
        if total_calls
        else 0
    )

    total_duration_seconds = sum(
    item["duration_seconds"]
    for item in results
)

    total_duration = format_duration(
        total_duration_seconds
    )


    trend = go.Figure()
    recent_for_chart = list(reversed(results[:10]))
    trend.add_trace(
        go.Scatter(
            x=[item["date"].strftime("%Y-%m-%d") for item in recent_for_chart],
            y=[score_percent(item) for item in recent_for_chart],
            mode="lines+markers",
            line={"width": 3},
        )
    )
    trend.update_layout(
        template="plotly_white",
        height=260,
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        yaxis={"range": [0, 100], "title": "Score"},
        showlegend=False,
    )

    
    recent_rows = [
        {
            "Call Name": item["recording"],
            "Type": item["call_type"],
            "Score": f"{score_percent(item):.1f}",
            "Duration": item["duration"],
            "Date": item["date"].strftime("%Y-%m-%d %H:%M"),
        }
        for item in results[:8]
    ]

    return page_shell(

        [
            html.Div(
                [
                    stat_card("Total Calls", str(total_calls), "Analyzed recordings"),
                    stat_card("Average Score", f"{avg_score:.1f}", "Out of 100"),
                    stat_card("Total Call Duration",total_duration, "Across analyzed calls"),
                    stat_card("Quality Rate", f"{quality_rate:.1f}%", "Calls scoring 85+"),
                ],
                className="stats-grid",
            ),
            html.Div(
    [
        html.Div(
            "Score Trend Over Time",
            className="section-title"
        ),
        dcc.Graph(
            figure=trend,
            config={
                "displayModeBar": False
            }
        ),
    ],
    className="panel",
),
            html.Div(
                [
                    html.Div("Recent Activity", className="section-title"),
                    dash_table.DataTable(
                        data=recent_rows,
                        columns=[{"name": key, "id": key} for key in (recent_rows[0].keys() if recent_rows else ["Call Name", "Type", "Score", "Duration", "Date"])],
                        page_size=8,
                        style_table={"overflowX": "auto"},
                        style_cell={"padding": "12px", "border": "none", "textAlign": "left"},
                        style_header={"fontWeight": "700", "backgroundColor": "#f7f8fa", "border": "none"},
                    ),
                ],
                className="panel",
            ),
        ]
    )


def analytics_page() -> Any:
    results = load_results()
    counts = {"Green": 0, "Amber": 0, "Red": 0, "Fail": 0}
    for item in results:
        score = score_percent(item)
        counts["Fail" if score == 0 else score_bucket(score)] += 1

    total = max(len(results), 1)
    failed_metrics: dict[str, int] = {}
    for call in results:
        for metric in call.get("metrics", []):
            if str(metric.get("result", "")).upper() == "FAIL":
                name = str(metric.get("name", "Unknown"))
                failed_metrics[name] = failed_metrics.get(name, 0) + 1

    bar = go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values())))
    bar.update_layout(template="plotly_white", height=270, margin={"l": 20, "r": 20, "t": 10, "b": 20})

    trend = go.Figure()
    recent = list(reversed(results[:10]))
    trend.add_trace(go.Scatter(
        x=[item["date"].strftime("%Y-%m-%d") for item in recent],
        y=[score_percent(item) for item in recent],
        mode="lines+markers",
    ))
    trend.update_layout(template="plotly_white", height=270, yaxis={"range": [0, 100]}, margin={"l": 20, "r": 20, "t": 10, "b": 20})

    failed_rows = [{"Metric": name, "Failures": count} for name, count in sorted(failed_metrics.items(), key=lambda item: item[1], reverse=True)]

    return page_shell(
        [
            html.Div(
                [
                    stat_card("Green", str(counts["Green"]), f"{counts['Green'] / total * 100:.1f}%"),
                    stat_card("Amber", str(counts["Amber"]), f"{counts['Amber'] / total * 100:.1f}%"),
                    stat_card("Red", str(counts["Red"]), f"{counts['Red'] / total * 100:.1f}%"),
                    stat_card("Fail", str(counts["Fail"]), f"{counts['Fail'] / total * 100:.1f}%"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    html.Div([html.Div("Score Distribution by Category", className="section-title"), dcc.Graph(figure=bar, config={"displayModeBar": False})], className="panel"),
                    html.Div([html.Div("Evaluation Trend", className="section-title"), dcc.Graph(figure=trend, config={"displayModeBar": False})], className="panel"),
                ],
                className="two-column",
            ),
            html.Div(
                [
                    html.Div("Top Failed Metrics", className="section-title"),
                    dash_table.DataTable(
                        data=failed_rows,
                        columns=[{"name": "Metric", "id": "Metric"}, {"name": "Failures", "id": "Failures"}],
                        page_size=10,
                        style_cell={"padding": "12px", "border": "none"},
                        style_header={"fontWeight": "700", "border": "none"},
                    ),
                ],
                className="panel",
            ),
        ]
    )


def batch_page() -> Any:
    return page_shell(
        [
            html.Div(
                [
                    html.Div("Audio Upload Area", className="section-title"),
                    dcc.Upload(
                        id="batch-upload",
                        multiple=True,
                        children=html.Div([
                            html.Div("Drop call recordings here", className="upload-title"),
                            html.Div("or click to browse · WAV · MP3 · M4A · FLAC", className="upload-subtitle"),
                        ]),
                        className="upload-box",
                    ),
                    html.Div(
                        [
                            html.Div([
                                html.Label("Auto Detect Call Type", className="field-label"),
                                dbc.Switch(id="auto-detect-call-type", value=True, label="Enabled"),
                            ]),
                            html.Div([
                                html.Label("Call Type", className="field-label"),
                                dcc.Dropdown(
                                    id="call-type",
                                    options=[
                                        {"label": "Customer Support", "value": "Customer Support"},
                                        {"label": "Sales", "value": "Sales"},
                                        {"label": "Complaint", "value": "Complaint"},
                                    ],
                                    value="Customer Support",
                                    clearable=False,
                                ),
                            ]),
                            html.Div([
                                html.Label("Parallel Workers", className="field-label"),
                                dcc.Dropdown(options=[1, 2, 3, 4], value=1, clearable=False),
                            ]),
                        ],
                        className="three-column",
                    ),
                    dbc.Button("START BATCH PROCESSING", id="start-batch", className="primary-btn"),
                    html.Div(id="batch-message", className="status-message"),
                ],
                className="panel",
            ),
            html.Div([
                html.Div("Queue", className="section-title"),
                dash_table.DataTable(
                    id="batch-table",
                    columns=[
                        {"name": "File Name", "id": "File Name"},
                        {"name": "Status", "id": "Status"},
                        {"name": "Progress", "id": "Progress"},
                        {"name": "Duration", "id": "Duration"},
                    ],
                    data=[],
                    style_cell={"padding": "12px", "border": "none"},
                    style_header={"fontWeight": "700", "border": "none"},
                ),
            ], className="panel"),
        ]
    )


def history_page() -> Any:
    results = load_results()
    if not results:
        return page_shell(html.Div("No analyzed calls yet.", className="empty-state panel"))

    calls = []
    for item in results:
        metric_rows = [
            {
                "Metric": metric.get("name", ""),
                "Result": metric.get("result", ""),
                "Score": metric.get("score", 0),
                "Weight": metric.get("weight", 0),
                "Type": metric.get("metric_type", ""),
                "Comments": metric.get("justification", ""),
            }
            for metric in item.get("metrics", [])
        ]
        transcript = "Transcript not found."
        if item["transcript_file"].exists():
            try:
                transcript = item["transcript_file"].read_text(encoding="utf-8")
            except OSError:
                pass

        calls.append(
            dbc.AccordionItem(
                [
                    html.Div(
                        [
                            stat_card("Analysis Score", f"{score_percent(item):.1f}", "Normalized"),
                            stat_card("Call Type", item["call_type"], ""),
                            stat_card("Analysis Count", "1", "Stored analysis"),
                        ],
                        className="stats-grid",
                    ),
                    html.Div("Evaluation Results", className="section-title"),
                    dash_table.DataTable(
                        data=metric_rows,
                        columns=[
                            {"name": key, "id": key}
                            for key in ["Metric", "Result", "Score", "Weight", "Type", "Comments"]
                        ],
                        style_cell={"padding": "10px", "whiteSpace": "normal", "height": "auto", "border": "none"},
                        style_header={"fontWeight": "700", "border": "none"},
                    ),
                    html.Div("Transcript", className="section-title"),
                    html.Pre(transcript, className="transcript-box"),
                    html.Div([
                        html.A("Download Analysis", href=f"/download/{item['recording']}_analysis.json", className="secondary-btn"),
                        html.A("Download Transcript", href=f"/download/{item['recording']}.txt", className="secondary-btn"),
                    ], className="button-row"),
                ],
                title=(
                    f"{item['recording']} · Score {score_percent(item):.1f} · "
                    f"{item['date'].strftime('%Y-%m-%d %H:%M')}"
                ),
            )
        )

    return page_shell(dbc.Accordion(calls, start_collapsed=True, always_open=False))


def metrics_page() -> Any:
    return page_shell(
        html.Div(
            [
                dcc.Tabs(
                    id="metrics-tabs",
                    value="view",
                    children=[
                        dcc.Tab(label="View Metrics", value="view"),
                        dcc.Tab(label="Add Metrics", value="add"),
                        dcc.Tab(label="Edit Metrics", value="edit"),
                        dcc.Tab(label="Import Metrics", value="import"),
                        dcc.Tab(label="SOP Upload", value="sop"),
                    ],
                ),
                html.Div(id="metrics-tab-content", className="tab-content"),
            ],
            className="panel",
        )
    )


def settings_page() -> Any:
    return page_shell(
        html.Div(
            [
                html.Div("Azure OpenAI Configuration", className="section-title"),
                html.Div([
                    html.Div([html.Label("Endpoint URL", className="field-label"), dbc.Input(id="azure-endpoint")]),
                    html.Div([html.Label("API Key", className="field-label"), dbc.Input(id="azure-api-key", type="password")]),
                    html.Div([html.Label("API Version", className="field-label"), dbc.Input(id="azure-api-version")]),
                    html.Div([html.Label("Transcription Model", className="field-label"), dbc.Input(value="Whisper")]),
                    html.Div([html.Label("Evaluation Model", className="field-label"), dbc.Input(value="Qwen 2.5 7B")]),
                    html.Div([html.Label("Embedding Model", className="field-label"), dbc.Input(value="all-MiniLM-L6-v2")]),
                ], className="two-column-form"),
                html.Div([
                    dbc.Button("TEST CONNECTION", id="test-connection", className="primary-btn"),
                    dbc.Button("SAVE CONFIGURATION", id="save-config", className="secondary-btn"),
                    dbc.Button("RESET", className="secondary-btn"),
                ], className="button-row"),
                html.Div(id="settings-message", className="status-message"),
            ],
            className="panel",
        )
    )


def sidebar() -> html.Div:
    items = [
        ("Dashboard", "dashboard"),
        ("Analytics", "analytics"),
        ("Batch Processing", "batch"),
        ("Call History", "history"),
        ("Metrics Management", "metrics"),
        ("Settings", "settings"),
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.Div("S", className="logo-mark"),
                    html.Div([
                        html.Div(APP_NAME, className="logo-text"),
                        html.Div("AI Call Quality Platform", className="logo-subtitle"),
                    ]),
                ],
                className="brand-block",
            ),
            html.Div([
                dbc.Button(
                    [html.Span("▦", className="nav-icon"), html.Span(label)],
                    id=f"nav-{key}",
                    className="nav-button",
                    color="link",
                )
                for label, key in items
            ], className="nav-list"),
            html.Div(
                [
                    html.Div("Environment", className="sidebar-caption"),
                    html.Div([html.Span(className="status-dot"), html.Span("Local / POC")], className="sidebar-status"),
                ],
                className="sidebar-footer",
            ),
        ],
        className="sidebar",
    )


def top_header() -> html.Div:
    return html.Div(
        [
            html.Div(id="page-title", className="page-title"),
            html.Div([
                html.Div([html.Span(className="status-dot"), html.Span("Azure: Connected")], className="connection-pill"),
                html.Div("User", className="user-pill"),
            ], className="header-right"),
        ],
        className="top-header",
    )


PAGE_BUILDERS = {
    "dashboard": ("Dashboard", dashboard_page),
    "analytics": ("Analytics", analytics_page),
    "batch": ("Batch Processing", batch_page),
    "history": ("Call History", history_page),
    "metrics": ("Metrics Management", metrics_page),
    "settings": ("Settings", settings_page),
}


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title=APP_NAME,
)

app.layout = html.Div(
    [
        sidebar(),
        html.Div(
            [
                top_header(),

                html.Div(
                    dashboard_page(),
                    id="page-content"
                ),
            ],
            className="main-area"
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("page-title", "children"),
    Output("page-content", "children"),
    [Input(f"nav-{key}", "n_clicks") for key in PAGE_BUILDERS],
)
def navigate(*_clicks: Any) -> tuple[str, Any]:

    triggered = dash.ctx.triggered_id

    if triggered is None:
        return "Dashboard", dashboard_page()

    key = triggered.replace(
        "nav-",
        "",
        1
    )

    if key not in PAGE_BUILDERS:
        return "Dashboard", dashboard_page()

    title, builder = PAGE_BUILDERS[key]

    return title, builder()



@app.callback(
    Output("metrics-tab-content", "children"),
    Input("metrics-tabs", "value")
)
def render_metrics_tab(tab: str) -> Any:

    metrics = load_metrics()

    if tab == "view":

        if not metrics:

            return html.Div(
                "No metrics configured.",
                className="empty-state"
            )

        rows = []

        for metric in metrics:

            rows.append(
                {
                    "Metric": metric["name"],
                    "Type": metric["metric_type"],
                    "Weight": metric["weight"],
                    "Criteria": " ".join(
                        metric["criteria"]
                    )
                }
            )

        return html.Div(
            [
                html.Div(
                    "Configured Metrics",
                    className="section-title"
                ),

                dash_table.DataTable(
                    data=rows,
                    columns=[
                        {
                            "name": "Metric",
                            "id": "Metric"
                        },
                        {
                            "name": "Type",
                            "id": "Type"
                        },
                        {
                            "name": "Weight",
                            "id": "Weight"
                        },
                        {
                            "name": "Criteria",
                            "id": "Criteria"
                        }
                    ],
                    page_size=10,
                    style_table={
                        "overflowX": "auto"
                    },
                    style_cell={
                        "padding": "12px",
                        "whiteSpace": "normal",
                        "height": "auto",
                        "border": "none",
                        "textAlign": "left"
                    },
                    style_header={
                        "fontWeight": "700",
                        "backgroundColor": "#f7f8fa",
                        "border": "none"
                    }
                )
            ]
        )

    if tab == "add":

        return html.Div(
            [
                html.Div(
                    "Add New Metric",
                    className="section-title"
                ),

                html.Label(
                    "Metric Name",
                    className="field-label"
                ),

                dbc.Input(
                    id="add-metric-name",
                    placeholder="Example: Email"
                ),

                html.Label(
                    "Metric Type",
                    className="field-label form-spacer"
                ),

                dcc.Dropdown(
                    id="add-metric-type",
                    options=[
                        {
                            "label": "Fatal",
                            "value": "Fatal"
                        },
                        {
                            "label": "Non-Fatal",
                            "value": "Non-Fatal"
                        }
                    ],
                    placeholder="Select metric type"
                ),

                html.Label(
                    "Weight",
                    className="field-label form-spacer"
                ),

                dbc.Input(
                    id="add-metric-weight",
                    type="number",
                    min=0,
                    step=0.5,
                    placeholder="Example: 5"
                ),

                html.Label(
                    "Criteria",
                    className="field-label form-spacer"
                ),

                dbc.Textarea(
                    id="add-metric-criteria",
                    placeholder="Example: Is email asked by the agent if possible to ask",
                    style={
                        "height": "120px"
                    }
                ),

                dbc.Button(
                    "SAVE METRIC",
                    id="save-metric",
                    className="primary-btn form-spacer"
                ),

                html.Div(
                    id="add-metric-message",
                    className="status-message"
                )
            ]
        )

    if tab == "edit":

        if not metrics:

            return html.Div(
                "No metrics available to edit.",
                className="empty-state"
            )

        options = []

        for index, metric in enumerate(metrics):

            options.append(
                {
                    "label": metric["name"],
                    "value": index
                }
            )

        return html.Div(
            [
                html.Div(
                    "Edit Metric",
                    className="section-title"
                ),

                html.Label(
                    "Select Metric",
                    className="field-label"
                ),

                dcc.Dropdown(
                    id="edit-metric-select",
                    options=options,
                    placeholder="Select a metric"
                ),

                html.Label(
                    "Metric Name",
                    className="field-label form-spacer"
                ),

                dbc.Input(
                    id="edit-metric-name"
                ),

                html.Label(
                    "Metric Type",
                    className="field-label form-spacer"
                ),

                dcc.Dropdown(
                    id="edit-metric-type",
                    options=[
                        {
                            "label": "Fatal",
                            "value": "Fatal"
                        },
                        {
                            "label": "Non-Fatal",
                            "value": "Non-Fatal"
                        }
                    ]
                ),

                html.Label(
                    "Weight",
                    className="field-label form-spacer"
                ),

                dbc.Input(
                    id="edit-metric-weight",
                    type="number",
                    min=0,
                    step=0.5
                ),

                html.Label(
                    "Criteria",
                    className="field-label form-spacer"
                ),

                dbc.Textarea(
                    id="edit-metric-criteria",
                    style={
                        "height": "120px"
                    }
                ),

                html.Div(
    [
        dbc.Button(
            "UPDATE METRIC",
            id="update-metric",
            className="primary-btn"
        ),

        dbc.Button(
            "DELETE METRIC",
            id="delete-metric",
            className="delete-btn"
        )
    ],
    className="button-row form-spacer"
),

                html.Div(
                    id="edit-metric-message",
                    className="status-message"
                )
            ]
        )

    if tab == "import":

        return html.Div(
            [
                dcc.Upload(
                    children=html.Div(
                        "Upload CSV, Excel or JSON"
                    ),
                    className="upload-box"
                ),

                dbc.Button(
                    "IMPORT",
                    className="primary-btn form-spacer"
                )
            ]
        )

    return html.Div(
        [
            dcc.Upload(
                children=html.Div(
                    "Upload PDF, DOCX or TXT SOP"
                ),
                className="upload-box"
            ),

            dbc.Button(
                "UPLOAD SOP",
                className="primary-btn form-spacer"
            )
        ]
    )

@app.callback(
    Output("add-metric-message", "children"),
    Input("save-metric", "n_clicks"),
    State("add-metric-name", "value"),
    State("add-metric-type", "value"),
    State("add-metric-weight", "value"),
    State("add-metric-criteria", "value"),
    prevent_initial_call=True
)
def save_metric(
    n_clicks: int | None,
    name: str | None,
    metric_type: str | None,
    weight: float | None,
    criteria: str | None
) -> str:

    if not n_clicks:
        return ""

    if not name or not metric_type or weight is None or not criteria:

        return "Please fill all metric fields."

    metrics = load_metrics()

    metric = {
        "name": name.strip(),
        "metric_type": metric_type,
        "criteria": [criteria.strip()],
        "weight": float(weight)
    }

    metrics.append(metric)

    save_metrics(metrics)

    return f"Metric '{name}' added successfully."

@app.callback(
    Output("edit-metric-name", "value"),
    Output("edit-metric-type", "value"),
    Output("edit-metric-weight", "value"),
    Output("edit-metric-criteria", "value"),
    Input("edit-metric-select", "value"),
    prevent_initial_call=True
)
def load_metric_for_edit(
    metric_index: int | None
) -> tuple:

    if metric_index is None:

        return "", None, None, ""

    metrics = load_metrics()

    if metric_index < 0 or metric_index >= len(metrics):

        return "", None, None, ""

    metric = metrics[metric_index]

    return (
        metric["name"],
        metric["metric_type"],
        metric["weight"],
        metric["criteria"][0]
    )

@app.callback(
    Output("edit-metric-message", "children"),
    Input("update-metric", "n_clicks"),
    State("edit-metric-select", "value"),
    State("edit-metric-name", "value"),
    State("edit-metric-type", "value"),
    State("edit-metric-weight", "value"),
    State("edit-metric-criteria", "value"),
    prevent_initial_call=True
)
def update_selected_metric(
    n_clicks: int | None,
    metric_index: int | None,
    name: str | None,
    metric_type: str | None,
    weight: float | None,
    criteria: str | None
) -> str:

    if not n_clicks:
        return ""

    if metric_index is None:

        return "Please select a metric first."

    if not name or not metric_type or weight is None or not criteria:

        return "Please fill all metric fields."

    metrics = load_metrics()

    if metric_index < 0 or metric_index >= len(metrics):

        return "Invalid metric selected."

    metrics[metric_index] = {
        "name": name.strip(),
        "metric_type": metric_type,
        "criteria": [criteria.strip()],
        "weight": float(weight)
    }

    save_metrics(metrics)

    return f"Metric '{name}' updated successfully."

@app.callback(
    Output("edit-metric-message", "children", allow_duplicate=True),
    Input("delete-metric", "n_clicks"),
    State("edit-metric-select", "value"),
    prevent_initial_call=True
)
def delete_selected_metric(
    n_clicks: int | None,
    metric_index: int | None
) -> str:

    if not n_clicks:
        return ""

    if metric_index is None:
        return "Please select a metric first."

    metrics = load_metrics()

    if metric_index < 0 or metric_index >= len(metrics):
        return "Invalid metric selected."

    deleted_metric = metrics.pop(
        metric_index
    )

    save_metrics(
        metrics
    )

    return (
        f"Metric '{deleted_metric['name']}' "
        f"deleted successfully."
    )



@app.callback(
    Output("batch-message", "children"),
    Input("start-batch", "n_clicks"),
    State("batch-upload", "filename"),
    prevent_initial_call=True,
)
def start_batch(n_clicks: int | None, filenames: list[str] | str | None) -> str:
    if not n_clicks:
        return ""
    if not filenames:
        return "Select at least one recording first."
    return "Batch UI is ready. Connect this callback to the existing orchestrator/backend pipeline."


@app.callback(
    Output("settings-message", "children"),
    Input("test-connection", "n_clicks"),
    prevent_initial_call=True,
)
def test_connection(n_clicks: int | None) -> str:
    if not n_clicks:
        return ""
    return "Connection test UI is ready."


if __name__ == "__main__":
    app.run(debug=True)
