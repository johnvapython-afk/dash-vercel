# app/index.py
import os
import io
from datetime import datetime

import numpy as np
import dash
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objs as go
import pandas as pd

DEFAULT_CLAIMS = "2701, 2799, 2626 "
DEFAULT_TRENDS = "0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7"
DEFAULT_VOLATILITY = 5
DEFAULT_SIMS = 1000
DEFAULT_SEED = 42
MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

dash_app = Dash(__name__)
dash_app.title = "Claims Projection"

dash_app.layout = html.Div(
    style={"fontFamily": "Segoe UI, Roboto, Arial", "padding": "16px"},
    children=[
        html.H2("FY26 Claims Projection _ Random Trend Simulation (Feb2026YTD)"),
        html.P("Provide observed monthly claims and monthly trend options (% per month)."),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
            children=[
                html.Div([
                    html.Label("Observed claims (comma-separated)"),
                    dcc.Input(id="claims-input", type="text", value=DEFAULT_CLAIMS, style={"width": "100%"}),
                    html.Small("Example: 120000, 115000, 130000, 125000"),
                ]),
                html.Div([
                    html.Label("Trend options (% per month, comma-separated)"),
                    dcc.Input(id="trends-input", type="text", value=DEFAULT_TRENDS, style={"width": "100%"}),
                    html.Small("Example: 0.5, 1.0, 1.5 (interpreted as +0.5% etc.)"),
                ]),
                html.Div([
                    html.Label("Monthly volatility (?, percent)"),
                    dcc.Slider(
                        id="vol-input", min=0.0, max=5.0, step=0.1, value=DEFAULT_VOLATILITY,
                        marks={0: "0%", 1: "1%", 2: "2%", 3: "3%", 4: "4%", 5: "5%"},
                    ),
                    html.Small("Noise ~ Normal(0, ?). Applied additively to the selected trend."),
                ]),
                html.Div([
                    html.Label("Number of simulations"),
                    dcc.Slider(
                        id="sims-input", min=100, max=5000, step=100, value=DEFAULT_SIMS,
                        marks={100: "100", 1000: "1k", 3000: "3k", 5000: "5k"},
                    ),
                ]),
                html.Div([
                    dcc.Input(id="seed-input", type="number", value=DEFAULT_SEED,
                              style={"width": "150px", "display": "none"}),
                ]),
                html.Div([
                    html.Label("Run"),
                    html.Button("Run Simulation", id="run-btn", n_clicks=0),
                ]),
                html.Div([
                    html.Label("Download"),
                    html.Button("Download CSV", id="download-btn", n_clicks=0, disabled=True),
                ]),
            ],
        ),
        html.Hr(),
        html.Div(
            style={"display": "flex", "gap": "24px", "flexWrap": "wrap"},
            children=[
                html.Div([html.H4("Mean annual total"), html.Div(id="mean-total", style={"fontSize": "26px"})]),
                html.Div([html.H4("Median annual total"), html.Div(id="median-total", style={"fontSize": "26px"})]),
                html.Div([html.H4("5th_95th pct"), html.Div(id="ci-total", style={"fontSize": "26px"})]),
                html.Div([html.H4("Observed months"), html.Div(id="obs-months", style={"fontSize": "26px"})]),
            ],
        ),
        dcc.Graph(id="timeseries-graph", style={"height": "48vh"}),
        dcc.Graph(id="timeseries-graph_accu", style={"height": "48vh"}),
        dcc.Graph(id="hist-graph", style={"height": "36vh"}),
        dcc.Download(id="download"),
        dcc.Store(id="sim-data"),
        dcc.Store(id="sim-meta"),
    ],
)

def parse_number_list(text: str):
    items = [s.strip() for s in (text or "").split(",") if s.strip() != ""]
    nums = []
    for x in items:
        try:
            nums.append(float(x))
        except ValueError:
            continue
    return nums

@dash_app.callback(
    Output("timeseries-graph_accu", "figure"),
    Output("timeseries-graph", "figure"),
    Output("hist-graph", "figure"),
    Output("mean-total", "children"),
    Output("median-total", "children"),
    Output("ci-total", "children"),
    Output("obs-months", "children"),
    Output("sim-data", "data"),         # results DF as JSON
    Output("sim-meta", "data"),         # inputs & metadata
    Output("download-btn", "disabled"), # enable after run
    Input("run-btn", "n_clicks"),
    State("claims-input", "value"),
    State("trends-input", "value"),
    State("vol-input", "value"),
    State("sims-input", "value"),
    State("seed-input", "value"),
)
def run_simulation(n_clicks, claims_text, trends_text, vol_pct, sims, seed):
    observed = parse_number_list(claims_text)
    trends_pct = parse_number_list(trends_text)  # percent per month
    vol_pct = float(vol_pct or 0.0)
    sims = int(sims or 1000)
    seed = None if seed is None else int(seed)

    # Guardrails & slice
    if len(observed) == 0: observed = [100000.0]
    if len(trends_pct) == 0: trends_pct = [0.0]
    obs_n = min(len(observed), 12)
    observed = observed[:obs_n]
    observed_acc = observed[:obs_n]
    remaining = max(0, 12 - obs_n)

    trend_options = np.array(trends_pct, dtype=float) / 100.0
    sigma = float(vol_pct) / 100.0
    if seed is not None: np.random.seed(seed)

    paths = np.zeros((sims, 12), dtype=float)
    paths_acc = np.zeros((sims, 12), dtype=float)

    # Observed months copied to all sims
    for i in range(obs_n):
        paths[:, i] = observed[i]
        if i == 0:
            observed_acc[i] = observed[i]; paths_acc[:, i] = observed[i]
        else:
            observed_acc[i] = observed_acc[i - 1] + observed[i]
            paths_acc[:, i] = paths_acc[:, i - 1] + observed[i]

    # Simulate remaining months: random trend + noise
    if remaining > 0:
        last = np.full(sims, sum(observed) / len(observed), dtype=float)
        for step in range(remaining):
            noise = np.random.normal(loc=0.0, scale=sigma, size=sims)
            picked_trends = np.random.choice(trend_options, size=sims, replace=True)
            growth = picked_trends + noise
            next_vals = np.maximum(0.0, last * (1.0 + growth))
            col = obs_n + step
            paths[:, col] = next_vals
            paths_acc[:, col] = paths_acc[:, col - 1] + next_vals
            last = next_vals

    # Stats
    annual_totals = paths.sum(axis=1)
    mean_total = float(np.mean(annual_totals))
    median_total = float(np.median(annual_totals))
    p5 = float(np.percentile(annual_totals, 5))
    p95 = float(np.percentile(annual_totals, 95))

    # Quantiles per month
    q5, q50, q95 = np.percentile(paths, [5, 50, 95], axis=0)

    # Accumulated quantiles
    q5_acc = np.zeros(12, dtype=float)
    q50_acc = np.zeros(12, dtype=float)
    q95_acc = np.zeros(12, dtype=float)
    for i in range(12):
        q5_acc[i]  = q5[i]  if i == 0 else q5_acc[i - 1]  + q5[i]
        q50_acc[i] = q50[i] if i == 0 else q50_acc[i - 1] + q50[i]
        q95_acc[i] = q95[i] if i == 0 else q95_acc[i - 1] + q95[i]

    x = MONTHS[:12]

    # Timeseries (monthly)
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=x[:obs_n], y=observed[:obs_n],
                                mode="lines+markers", line=dict(color="black", width=2), name="Observed"))
    fig_ts.add_trace(go.Scatter(x=x, y=q50, mode="lines", line=dict(color="#1f77b4", width=3), name="Median"))
    fig_ts.add_trace(go.Scatter(x=x, y=q5, mode="lines", line=dict(color="#1f77b4", width=0), showlegend=False))
    fig_ts.add_trace(go.Scatter(x=x, y=q95, mode="lines", line=dict(color="#1f77b4", width=0),
                                fill="tonexty", fillcolor="rgba(31,119,180,0.20)", name="5–95% band"))
    if obs_n < 12:
        fig_ts.add_vline(x=obs_n - 0.5, line=dict(color="gray", dash="dot"),
                         annotation_text="Forecast begins", annotation_position="top right")
    fig_ts.update_layout(title="Monthly Claims — Observed & Simulated", xaxis_title="Month",
                         yaxis_title="Claims", template="plotly_white",
                         margin=dict(l=40, r=20, t=50, b=40),
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.5))

    # Timeseries (accumulative)
    fig_ts_acc = go.Figure()
    fig_ts_acc.add_trace(go.Scatter(x=x[:obs_n], y=observed_acc[:obs_n],
                                    mode="lines+markers", line=dict(color="black", width=2), name="Observed"))
    fig_ts_acc.add_trace(go.Scatter(x=x, y=q50_acc, mode="lines", line=dict(color="#1f77b4", width=3), name="Median"))
    fig_ts_acc.add_trace(go.Scatter(x=x, y=q5_acc, mode="lines", line=dict(color="#1f77b4", width=0), showlegend=False))
    fig_ts_acc.add_trace(go.Scatter(x=x, y=q95_acc, mode="lines", line=dict(color="#1f77b4", width=0),
                                    fill="tonexty", fillcolor="rgba(31,119,180,0.20)", name="5–95% band"))
    if obs_n < 12:
        fig_ts_acc.add_vline(x=obs_n - 0.5, line=dict(color="gray", dash="dot"),
                             annotation_text="Forecast begins", annotation_position="top right")
    fig_ts_acc.update_layout(title="Accumulative Monthly Claims — Observed & Simulated", xaxis_title="Month",
                             yaxis_title="Claims", template="plotly_white",
                             margin=dict(l=40, r=20, t=50, b=40),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.5))

    # Histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=annual_totals, nbinsx=10, marker_color="#2ca02c", opacity=0.85))
    fig_hist.add_vline(x=mean_total, line_color="black", line_dash="dash",
                       annotation_text=f"Mean: {mean_total:,.0f}", annotation_position="top right")
    fig_hist.add_vline(x=median_total, line_color="#d62728", line_dash="dot",
                       annotation_text=f"Median: {median_total:,.0f}", annotation_position="top right")
    fig_hist.update_layout(title="Distribution of Annual Totals (Simulated)", xaxis_title="Annual total",
                           yaxis_title="Count", template="plotly_white",
                           margin=dict(l=40, r=20, t=50, b=40), bargap=0.02)

    # Build exportable DF & metadata
    df_export = pd.DataFrame({"sim_id": np.arange(sims, dtype=int)})
    for i, name in enumerate(MONTHS[:12]):
        df_export[name] = paths[:, i]
    df_export["annual_total"] = annual_totals
    df_export["observed_months"] = obs_n
    df_export["volatility_pct"] = vol_pct
    df_export["sims"] = sims
    export_json = df_export.to_json(orient="split")

    meta = {
        "observed_input_text": claims_text or "",
        "observed_parsed": observed,
        "trend_input_text": trends_text or "",
        "trend_options_pct": [float(x) for x in trends_pct],
        "volatility_pct": vol_pct,
        "sims": sims,
        "seed": seed,
        "months_labels": MONTHS[:12],
        "observed_months_count": obs_n,
    }

    # Enable the Download button
    return (fig_ts_acc, fig_ts, fig_hist,
            f"{mean_total:,.0f}", f"{median_total:,.0f}", f"{p5:,.0f} _ {p95:,.0f}",
            f"{obs_n} month(s)",
            export_json, meta, False)

@dash_app.callback(
    Output("download", "data"),
    Input("download-btn", "n_clicks"),
    State("sim-data", "data"),
    State("sim-meta", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, sim_json, meta):
    if not sim_json or not meta:
        return dash.no_update

    df = pd.read_json(sim_json, orient="split")

    buf = io.StringIO()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    header_lines = [
        "Parameter,Value",
        f"Generated At,{ts}",
        f"Observed Claims (raw),\"{meta.get('observed_input_text','')}\"",
        f"Observed Claims (parsed),\"{','.join(str(x) for x in meta.get('observed_parsed', []))}\"",
        f"Trend Options % (raw),\"{meta.get('trend_input_text','')}\"",
        f"Trend Options % (parsed),\"{','.join(str(x) for x in meta.get('trend_options_pct', []))}\"",
        f"Volatility %,{meta.get('volatility_pct','')}",
        f"Simulations,{meta.get('sims','')}",
        f"Seed,{meta.get('seed','')}",
        f"Observed Months Count,{meta.get('observed_months_count','')}",
        f"Month Labels,\"{','.join(meta.get('months_labels', []))}\"",
    ]
    buf.write("\n".join(header_lines))
    buf.write("\n\n")
    df.to_csv(buf, index=False)
    return dcc.send_string(buf.getvalue(), "claims_simulation_with_inputs.csv")

# WSGI app for Vercel (required)
app = dash_app.server

# Local dev
if __name__ == "__main__":
    dash_app.run(debug=True)