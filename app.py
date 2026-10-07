from __future__ import annotations

import base64
import binascii
import dash
import json
import logging
import os
import re
import webbrowser

from datetime import datetime
from pathlib import Path
from threading import RLock, Thread, Timer
from typing import Any
from uuid import uuid4

from dash import Dash, Input, Output, State, dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from metric_manager import load_metrics, save_metrics
from transcriber import transcribe_audio, save_transcript
from transcript_index import build_transcript_index
from evaluator import evaluate_transcript





APP_NAME = "Colleague Assist"
RESULTS_DIR = Path("results")
TRANSCRIPTS_DIR = Path("transcripts")
UPLOADS_DIR = Path("uploads")

ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Batch processing state
# ---------------------------------------------------------
# This is intentionally kept in memory for the local/POC Dash app.
# The UI polls this state every second while the worker processes
# recordings sequentially.
BATCH_STATE_LOCK = RLock()

BATCH_STATE: dict[str, Any] = {
    "running": False,
    "message": "",
    "jobs": [],
}


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
                    
                    "call_type": data.get(
                        "call_type",
                        "Customer Support",
                    ),
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


def _get_batch_jobs_snapshot() -> list[dict[str, Any]]:
    """
    Return a safe copy of the current batch queue.
    """
    with BATCH_STATE_LOCK:
        return [
            dict(job)
            for job in BATCH_STATE["jobs"]
        ]


def get_batch_rows() -> list[dict[str, str]]:
    """
    Convert the internal batch state into rows for the Dash table.
    """
    jobs = _get_batch_jobs_snapshot()

    return [
        {
            "File Name": str(job["filename"]),
            "Status": str(job["status"]),
            "Progress": f"{int(job.get('progress', 0))}%",
            "Duration": str(job.get("duration", "—")),
        }
        for job in jobs
    ]


def _update_batch_job(
    job_id: str,
    **changes: Any,
) -> None:
    """
    Update one job safely inside the batch queue.
    """
    with BATCH_STATE_LOCK:
        for job in BATCH_STATE["jobs"]:
            if job["job_id"] == job_id:
                job.update(changes)
                return


def _save_uploaded_audio(
    contents: str,
    filename: str,
) -> Path:
    """
    Decode a Dash upload and save it into the local uploads folder.

    The original filename is kept where possible. If a file with the
    same name already exists, a UUID suffix is added to avoid collision.
    """
    if not contents or "," not in contents:
        raise ValueError("Invalid uploaded file data.")

    safe_filename = Path(filename).name
    extension = Path(safe_filename).suffix.lower()

    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension or 'unknown'}. "
            "Allowed: WAV, MP3, M4A, FLAC."
        )

    header, encoded_data = contents.split(",", 1)

    if not header.startswith("data:"):
        raise ValueError("Invalid upload format.")

    try:
        file_bytes = base64.b64decode(
            encoded_data,
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise ValueError(
            "Could not decode the uploaded file."
        ) from exc

    UPLOADS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stored_path = UPLOADS_DIR / safe_filename

    if stored_path.exists():
        stem = stored_path.stem
        stored_path = (
            UPLOADS_DIR
            / f"{stem}_{uuid4().hex[:8]}{extension}"
        )

    stored_path.write_bytes(file_bytes)

    return stored_path


def _attach_batch_metadata(
    result_path: str | Path,
    call_type: str,
) -> None:
    """
    Store the call type selected in the UI inside the generated
    analysis JSON. The evaluator itself does not use call type.
    """
    path = Path(result_path)

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        data["call_type"] = call_type

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
            )

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        logger.exception(
            "Could not attach batch metadata to %s",
            path,
        )


def _process_one_batch_job(
    job: dict[str, Any],
) -> None:
    """
    Process exactly one recording using the existing backend pipeline.

    Flow:
        audio
          -> transcribe
          -> save transcript
          -> build FAISS index
          -> evaluate transcript
          -> save result JSON
    """
    job_id = str(job["job_id"])
    audio_path = Path(job["saved_path"])

    try:
        # ---------------------------------------------
        # STEP 1: TRANSCRIPTION
        # ---------------------------------------------
        _update_batch_job(
            job_id,
            status="Transcribing",
            progress=20,
        )

        logger.info(
            "Transcribing batch recording: %s",
            audio_path,
        )

        segments = transcribe_audio(
            str(audio_path)
        )

        transcript_path = save_transcript(
            segments,
            str(audio_path),
        )

        # ---------------------------------------------
        # STEP 2: INDEX + EVALUATION
        # ---------------------------------------------
        _update_batch_job(
            job_id,
            status="Evaluating",
            progress=60,
            transcript_path=str(transcript_path),
        )

        logger.info(
            "Building transcript index: %s",
            transcript_path,
        )

        build_transcript_index(
            transcript_path
        )

        logger.info(
            "Evaluating transcript: %s",
            transcript_path,
        )

        result_path = evaluate_transcript(
            transcript_path
        )

        _attach_batch_metadata(
            result_path,
            str(
                job.get(
                    "call_type",
                    "Customer Support",
                )
            ),
        )

        # ---------------------------------------------
        # STEP 3: COMPLETED
        # ---------------------------------------------
        duration_seconds = get_transcript_duration(
            Path(transcript_path)
        )

        _update_batch_job(
            job_id,
            status="Completed",
            progress=100,
            duration=format_duration(
                duration_seconds
            ),
            duration_seconds=duration_seconds,
            result_path=str(result_path),
        )

        logger.info(
            "Batch recording completed: %s -> %s",
            audio_path.name,
            result_path,
        )

    except Exception as exc:
        logger.exception(
            "Batch recording failed: %s",
            audio_path,
        )

        _update_batch_job(
            job_id,
            status="Failed",
            progress=0,
            duration="—",
            error=str(exc),
        )


def _run_batch_queue() -> None:
    """
    Process all currently queued recordings strictly one at a time.
    """
    with BATCH_STATE_LOCK:
        queue = [
            dict(job)
            for job in BATCH_STATE["jobs"]
            if job.get("status") == "Queued"
        ]

    completed = 0
    failed = 0

    try:
        for job in queue:
            _process_one_batch_job(job)

            with BATCH_STATE_LOCK:
                matching_job = next(
                    (
                        item
                        for item in BATCH_STATE["jobs"]
                        if item["job_id"] == job["job_id"]
                    ),
                    None,
                )

                if matching_job and matching_job.get("status") == "Completed":
                    completed += 1
                else:
                    failed += 1

                BATCH_STATE["message"] = (
                    f"Processed {completed + failed} of {len(queue)} recordings "
                    f"({completed} completed, {failed} failed)."
                )

    finally:
        with BATCH_STATE_LOCK:
            BATCH_STATE["running"] = False
            BATCH_STATE["message"] = (
                "Batch processing completed. "
                f"{completed} completed, {failed} failed."
            )


def _start_batch_thread() -> None:
    """
    Start the queue in a background thread so the Dash request is not blocked.
    """
    worker = Thread(
        target=_run_batch_queue,
        name="batch-processing-worker",
        daemon=True,
    )

    worker.start()


def batch_page() -> Any:
    return page_shell(
        [
            # Refresh the queue table every second while processing.
            dcc.Interval(
                id="batch-refresh",
                interval=1000,
                n_intervals=0,
            ),

            html.Div(
                [
                    html.Div(
                        "Audio Upload Area",
                        className="section-title",
                    ),

                    dcc.Upload(
                        id="batch-upload",
                        multiple=True,
                        accept=".wav,.mp3,.m4a,.flac",
                        children=html.Div(
                            [
                                html.Div(
                                    "Drop call recordings here",
                                    className="upload-title",
                                ),
                                html.Div(
                                    "or click to browse · WAV · MP3 · M4A · FLAC",
                                    className="upload-subtitle",
                                ),
                            ]
                        ),
                        className="upload-box",
                    ),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Auto Detect Call Type",
                                        className="field-label",
                                    ),
                                    dbc.Switch(
                                        id="auto-detect-call-type",
                                        value=True,
                                        label="Enabled",
                                    ),
                                ]
                            ),

                            html.Div(
                                [
                                    html.Label(
                                        "Call Type",
                                        className="field-label",
                                    ),
                                    dcc.Dropdown(
                                        id="call-type",
                                        options=[
                                            {
                                                "label": "Customer Support",
                                                "value": "Customer Support",
                                            },
                                            {
                                                "label": "Sales",
                                                "value": "Sales",
                                            },
                                            {
                                                "label": "Complaint",
                                                "value": "Complaint",
                                            },
                                        ],
                                        value="Customer Support",
                                        clearable=False,
                                    ),
                                ]
                            ),

                            html.Div(
                                [
                                    html.Label(
                                        "Processing Workers",
                                        className="field-label",
                                    ),
                                    dcc.Dropdown(
                                        id="batch-workers",
                                        options=[
                                            {
                                                "label": "1 worker",
                                                "value": 1,
                                            },
                                        ],
                                        value=1,
                                        clearable=False,
                                        disabled=True,
                                    ),
                                    html.Div(
                                        "Sequential processing only",
                                        className="field-hint",
                                    ),
                                ]
                            ),
                        ],
                        className="three-column",
                    ),

                    dbc.Button(
                        "START BATCH PROCESSING",
                        id="start-batch",
                        className="primary-btn",
                    ),

                    html.Div(
                        id="batch-message",
                        className="status-message",
                    ),
                ],
                className="panel",
            ),

            html.Div(
                [
                    html.Div(
                        "Queue",
                        className="section-title",
                    ),
                    dash_table.DataTable(
                        id="batch-table",
                        columns=[
                            {
                                "name": "File Name",
                                "id": "File Name",
                            },
                            {
                                "name": "Status",
                                "id": "Status",
                            },
                            {
                                "name": "Progress",
                                "id": "Progress",
                            },
                            {
                                "name": "Duration",
                                "id": "Duration",
                            },
                        ],
                        data=get_batch_rows(),
                        style_table={
                            "overflowX": "auto",
                        },
                        style_cell={
                            "padding": "12px",
                            "border": "none",
                        },
                        style_header={
                            "fontWeight": "700",
                            "border": "none",
                        },
                    ),
                ],
                className="panel",
            ),
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
    Output("batch-table", "data"),
    Output("batch-message", "children"),
    Output("start-batch", "disabled"),
    Input("batch-upload", "contents"),
    Input("start-batch", "n_clicks"),
    Input("batch-refresh", "n_intervals"),
    State("batch-upload", "filename"),
    State("call-type", "value"),
    State("auto-detect-call-type", "value"),
    State("batch-workers", "value"),
    prevent_initial_call=True,
)
def batch_controller(
    contents: list[str] | str | None,
    n_clicks: int | None,
    n_intervals: int | None,
    filenames: list[str] | str | None,
    call_type: str | None,
    auto_detect: bool | None,
    workers: int | None,
) -> tuple[list[dict[str, str]], str, bool]:
    """
    Handle file uploads, batch start, and live queue refresh.

    The actual audio processing happens in a background thread so that
    the browser can continue receiving queue updates while each recording
    is processed sequentially.
    """
    del n_clicks, n_intervals, workers

    triggered = dash.ctx.triggered_id

    # =========================================================
    # 1. FILE UPLOAD
    # =========================================================
    if triggered == "batch-upload":
        if not contents or not filenames:
            return (
                get_batch_rows(),
                "",
                BATCH_STATE["running"],
            )

        with BATCH_STATE_LOCK:
            if BATCH_STATE["running"]:
                return (
                    get_batch_rows(),
                    "A batch is already running. Please wait for it to finish.",
                    True,
                )

        content_list = (
            contents
            if isinstance(contents, list)
            else [contents]
        )

        filename_list = (
            filenames
            if isinstance(filenames, list)
            else [filenames]
        )

        added_jobs: list[dict[str, Any]] = []
        rejected_files: list[str] = []

        for file_contents, filename in zip(
            content_list,
            filename_list,
        ):
            try:
                saved_path = _save_uploaded_audio(
                    file_contents,
                    str(filename),
                )

                added_jobs.append(
                    {
                        "job_id": uuid4().hex,
                        "filename": Path(str(filename)).name,
                        "saved_path": str(saved_path),
                        "status": "Queued",
                        "progress": 0,
                        "duration": "—",
                        "duration_seconds": 0.0,
                        "call_type": call_type or "Customer Support",
                        "auto_detect": bool(auto_detect),
                    }
                )

            except (OSError, ValueError) as exc:
                rejected_files.append(
                    f"{Path(str(filename)).name}: {exc}"
                )

        with BATCH_STATE_LOCK:
            BATCH_STATE["jobs"].extend(added_jobs)

            if added_jobs and rejected_files:
                BATCH_STATE["message"] = (
                    f"{len(added_jobs)} recording(s) added to the queue. "
                    f"Rejected: {'; '.join(rejected_files)}"
                )
            elif added_jobs:
                BATCH_STATE["message"] = (
                    f"{len(added_jobs)} recording(s) added to the queue. "
                    "Processing will remain strictly sequential."
                )
            elif rejected_files:
                BATCH_STATE["message"] = (
                    "No recordings were added. "
                    f"Rejected: {'; '.join(rejected_files)}"
                )
            else:
                BATCH_STATE["message"] = "No recordings were added."

            running = bool(BATCH_STATE["running"])
            message = str(BATCH_STATE["message"])

        return (
            get_batch_rows(),
            message,
            running,
        )

    # =========================================================
    # 2. START BATCH
    # =========================================================
    if triggered == "start-batch":
        with BATCH_STATE_LOCK:
            if BATCH_STATE["running"]:
                return (
                    get_batch_rows(),
                    "Batch processing is already running.",
                    True,
                )

            queued_jobs = [
                job
                for job in BATCH_STATE["jobs"]
                if job.get("status") == "Queued"
            ]

            if not queued_jobs:
                message = "Select at least one queued recording first."
                BATCH_STATE["message"] = message
                return (
                    get_batch_rows(),
                    message,
                    False,
                )

            BATCH_STATE["running"] = True
            BATCH_STATE["message"] = (
                f"Batch processing started for {len(queued_jobs)} recording(s). "
                "One recording will be processed at a time."
            )
            message = str(BATCH_STATE["message"])

        _start_batch_thread()

        return (
            get_batch_rows(),
            message,
            True,
        )

    # =========================================================
    # 3. PERIODIC UI REFRESH
    # =========================================================
    with BATCH_STATE_LOCK:
        message = str(BATCH_STATE["message"])
        running = bool(BATCH_STATE["running"])

    return (
        get_batch_rows(),
        message,
        running,
    )


@app.callback(
    Output("settings-message", "children"),
    Input("test-connection", "n_clicks"),
    prevent_initial_call=True,
)
def test_connection(n_clicks: int | None) -> str:
    if not n_clicks:
        return ""
    return "Connection test UI is ready."

def open_browser() -> None:
    webbrowser.open_new("http://127.0.0.1:8050/")


if __name__ == "__main__":
    Timer(1, open_browser).start()

    host = os.getenv("DASH_HOST", "127.0.0.1")
    port = int(os.getenv("DASH_PORT", "8050"))
    debug = os.getenv("DASH_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
