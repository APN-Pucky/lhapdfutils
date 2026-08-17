from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import lhapdf
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

lhapdf.setVerbosity(0)

DEFAULT_SET = "BFG_II"
DEFAULT_MEMBER = 0
DEFAULT_PID = 2
DEFAULT_Z_POINTS = 1000
DEFAULT_Q_POINTS = 1000
APP_BG = "#f6f0e8"
PANEL_BG = "#fffaf5"
PLOT_BG = "#fffdf9"
GRID_COLOR = "#e8dccd"
TEXT_COLOR = "#2a221c"
ACCENT = "#c16926"
LINE_A = "#0f766e"
LINE_B = "#1d4ed8"
MARKER = "#ea580c"
PID_LABELS = {
    -15: "taubar",
    -13: "mubar",
    -11: "ebar",
    -5: "bbar",
    -4: "cbar",
    -3: "sbar",
    -2: "ubar",
    -1: "dbar",
    1: "d",
    2: "u",
    3: "s",
    4: "c",
    5: "b",
    11: "e",
    13: "mu",
    15: "tau",
    21: "g",
    22: "gamma",
}
PREFERRED_PIDS = [2, 1, 3, 21, -2, -1, -3, 4, 5, -4, -5, 22]


def available_sets() -> list[str]:
    names = sorted(lhapdf.availablePDFSets())
    if DEFAULT_SET in names:
        names.remove(DEFAULT_SET)
        names.insert(0, DEFAULT_SET)
    return names


SET_OPTIONS = available_sets()
INITIAL_SET = SET_OPTIONS[0] if SET_OPTIONS else None


def format_pid(pid: int) -> str:
    label = PID_LABELS.get(pid, f"PID {pid}")
    return f"{label} ({pid})"


def preferred_pid(flavors: list[int], current: int | None = None) -> int | None:
    if current in flavors:
        return current
    for pid in PREFERRED_PIDS:
        if pid in flavors:
            return pid
    return flavors[0] if flavors else None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def sanitize_member(value: Any, max_member: int) -> int:
    if max_member <= 0:
        return 0
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(max_member, parsed))


def sanitize_positive(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed) or parsed <= 0:
        return fallback
    return parsed


def sanitize_int(value: Any, fallback: int, minimum: int = 2, maximum: int = 300) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, parsed))


def sanitize_range(lower: Any, upper: Any, fallback_lower: float, fallback_upper: float) -> tuple[float, float]:
    lo = sanitize_positive(lower, fallback_lower)
    hi = sanitize_positive(upper, fallback_upper)
    if lo >= hi:
        hi = max(fallback_upper, lo * 1.01)
        if lo >= hi:
            lo = fallback_lower
            hi = fallback_upper
    return lo, hi


def midpoint(lower: float, upper: float, log_scale: bool) -> float:
    if log_scale:
        return 10 ** ((math.log10(lower) + math.log10(upper)) / 2.0)
    return (lower + upper) / 2.0


def encode_slider(value: float, log_scale: bool) -> float:
    return math.log10(value) if log_scale else value


def decode_slider(value: Any, log_scale: bool, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if log_scale:
        return 10**parsed
    return parsed


def format_number(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000 or magnitude < 0.01:
        return f"{value:.2e}"
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.3g}"


def make_marks(lower: float, upper: float, log_scale: bool) -> dict[float, str]:
    marks: dict[float, str] = {}
    if log_scale:
        start = math.floor(math.log10(lower))
        end = math.ceil(math.log10(upper))
        for exponent in range(start, end + 1):
            raw = 10**exponent
            if lower <= raw <= upper:
                marks[float(exponent)] = format_number(raw)
        marks[math.log10(lower)] = format_number(lower)
        marks[math.log10(upper)] = format_number(upper)
        return dict(sorted(marks.items()))
    step = (upper - lower) / 4.0
    for idx in range(5):
        value = lower + idx * step
        marks[round(value, 8)] = format_number(value)
    return marks


def slider_step(lower: float, upper: float, log_scale: bool) -> float:
    encoded_lower = encode_slider(lower, log_scale)
    encoded_upper = encode_slider(upper, log_scale)
    span = max(encoded_upper - encoded_lower, 1e-6)
    return span / 500.0


def quantity_label(quantity: str) -> str:
    return "z D(z, Q)" if quantity == "xfx" else "D(z, Q)"


def evaluate_quantity(pdf: Any, pid: int, z: float, q: float, quantity: str) -> float:
    raw = float(pdf.xfxQ(pid, z, q))
    if quantity == "xfx":
        return raw
    return raw / z


def axis_values(lower: float, upper: float, points: int, log_scale: bool) -> np.ndarray:
    if log_scale:
        return np.geomspace(lower, upper, points)
    return np.linspace(lower, upper, points)


def transform_surface(surface: np.ndarray, color_mode: str, quantity: str) -> tuple[np.ndarray, str]:
    if color_mode == "linear":
        return surface, quantity_label(quantity)
    if color_mode == "log":
        transformed = np.full_like(surface, np.nan, dtype=float)
        positive = surface > 0
        transformed[positive] = np.log10(surface[positive])
        return transformed, "log10(value)"

    finite = surface[np.isfinite(surface)]
    if finite.size == 0:
        return surface, "symlog(value)"
    reference = np.nanpercentile(np.abs(finite), 15)
    linthresh = max(reference, 1e-12)
    transformed = np.sign(surface) * np.log10(1.0 + np.abs(surface) / linthresh)
    return transformed, f"symlog(value / {linthresh:.1e})"


def base_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PLOT_BG,
        font={"family": "Avenir Next, Gill Sans, Trebuchet MS, sans-serif", "color": TEXT_COLOR},
        margin={"l": 56, "r": 20, "t": 64, "b": 52},
    )
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = base_figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        font={"size": 16, "color": TEXT_COLOR},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


@lru_cache(maxsize=64)
def get_pdf_set(set_name: str) -> Any:
    return lhapdf.getPDFSet(set_name)


@lru_cache(maxsize=256)
def get_pdf(set_name: str, member: int) -> Any:
    return lhapdf.mkPDF(set_name, member)


@lru_cache(maxsize=256)
def dataset_metadata(set_name: str, member: int) -> dict[str, Any]:
    pdf_set = get_pdf_set(set_name)
    pdf = get_pdf(set_name, member)
    z_min = max(float(pdf.xMin), 1e-12)
    z_max = min(float(pdf.xMax), 1.0)
    q_min = max(math.sqrt(float(pdf.q2Min)), 1e-9)
    q_max = math.sqrt(float(pdf.q2Max))
    return {
        "member_count": int(pdf_set.size),
        "description": str(pdf_set.description or "").strip(),
        "flavors": list(pdf.flavors()),
        "z_min": z_min,
        "z_max": z_max,
        "q_min": q_min,
        "q_max": q_max,
    }


@lru_cache(maxsize=96)
def sample_surface(
    set_name: str,
    member: int,
    pid: int,
    quantity: str,
    z_min: float,
    z_max: float,
    q_min: float,
    q_max: float,
    z_points: int,
    q_points: int,
    log_z: bool,
    log_q: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pdf = get_pdf(set_name, member)
    z_axis = axis_values(z_min, z_max, z_points, log_z)
    q_axis = axis_values(q_min, q_max, q_points, log_q)
    grid = np.empty((q_points, z_points), dtype=float)

    for row, q_value in enumerate(q_axis):
        grid[row, :] = [evaluate_quantity(pdf, pid, z_value, q_value, quantity) for z_value in z_axis]

    return z_axis, q_axis, grid


@lru_cache(maxsize=192)
def sample_z_projection(
    set_name: str,
    member: int,
    pid: int,
    quantity: str,
    z_min: float,
    z_max: float,
    q_value: float,
    z_points: int,
    log_z: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pdf = get_pdf(set_name, member)
    z_axis = axis_values(z_min, z_max, z_points, log_z)
    values = np.array([evaluate_quantity(pdf, pid, z_value, q_value, quantity) for z_value in z_axis], dtype=float)
    return z_axis, values


@lru_cache(maxsize=192)
def sample_q_projection(
    set_name: str,
    member: int,
    pid: int,
    quantity: str,
    q_min: float,
    q_max: float,
    z_value: float,
    q_points: int,
    log_q: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pdf = get_pdf(set_name, member)
    q_axis = axis_values(q_min, q_max, q_points, log_q)
    values = np.array([evaluate_quantity(pdf, pid, z_value, q_value, quantity) for q_value in q_axis], dtype=float)
    return q_axis, values


def heatmap_figure(
    set_name: str,
    member: int,
    pid: int,
    quantity: str,
    color_mode: str,
    z_min: float,
    z_max: float,
    q_min: float,
    q_max: float,
    z_points: int,
    q_points: int,
    log_z: bool,
    log_q: bool,
    z_star: float,
    q_star: float,
    interpolation: str,
    show_cell_grid: bool,
) -> go.Figure:
    z_axis, q_axis, surface = sample_surface(
        set_name,
        member,
        pid,
        quantity,
        z_min,
        z_max,
        q_min,
        q_max,
        z_points,
        q_points,
        log_z,
        log_q,
    )
    transformed, colorbar_title = transform_surface(surface, color_mode, quantity)
    fig = base_figure()

    if np.isfinite(transformed).any():
        fig.add_trace(
            go.Heatmap(
                x=z_axis,
                y=q_axis,
                z=transformed,
                customdata=surface,
                colorscale="Viridis",
                colorbar={"title": colorbar_title},
                hovertemplate="z=%{x:.4g}<br>Q=%{y:.4g}<br>value=%{customdata:.4e}<extra></extra>",
                xgap=1 if show_cell_grid else 0,
                ygap=1 if show_cell_grid else 0,
                zsmooth="best" if interpolation == "smooth" else False,
            )
        )
    else:
        fig.add_annotation(
            text="No finite values available for the selected color transform.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            font={"size": 16},
        )

    selected_value = evaluate_quantity(get_pdf(set_name, member), pid, z_star, q_star, quantity)
    fig.add_trace(
        go.Scatter(
            x=[z_star],
            y=[q_star],
            mode="markers",
            name="Selection",
            customdata=[[selected_value]],
            marker={"size": 13, "color": MARKER, "line": {"color": PANEL_BG, "width": 2}},
            hovertemplate="z*=%{x:.4g}<br>Q*=%{y:.4g}<br>value=%{customdata[0]:.4e}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"{quantity_label(quantity)} heatmap<br><sup>{set_name}, member {member}, {format_pid(pid)}</sup>",
        height=640,
    )
    fig.update_xaxes(title="z", type="log" if log_z else "linear", showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(title="Q", type="log" if log_q else "linear", showgrid=True, gridcolor=GRID_COLOR)
    return fig


def projection_figure(
    x_axis: np.ndarray,
    values: np.ndarray,
    x_star: float,
    y_star: float,
    title: str,
    x_title: str,
    y_title: str,
    x_log: bool,
    line_color: str,
) -> go.Figure:
    fig = base_figure()
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=values,
            mode="lines",
            line={"color": line_color, "width": 3},
            hovertemplate=f"{x_title}=%{{x:.4g}}<br>{y_title}=%{{y:.4e}}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_star],
            y=[y_star],
            mode="markers",
            marker={"size": 11, "color": MARKER, "line": {"color": PANEL_BG, "width": 2}},
            hovertemplate=f"{x_title}*=%{{x:.4g}}<br>{y_title}=%{{y:.4e}}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(title=title, height=320)
    fig.update_xaxes(title=x_title, type="log" if x_log else "linear", showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(title=y_title, showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


def build_app() -> Dash:
    app = Dash(__name__, title="LHAPDF Explorer")

    if not SET_OPTIONS:
        app.layout = html.Div(
            className="empty-state",
            children=[
                html.H1("LHAPDF Explorer"),
                html.P("No LHAPDF sets were found in the current environment."),
            ],
        )
        return app

    app.layout = html.Div(
        className="page-shell",
        children=[
            html.Div(
                className="app-shell",
                children=[
                    html.Aside(
                        className="control-panel",
                        children=[
                            html.Div(
                                className="panel-header",
                                children=[
                                    html.H1("LHAPDF Explorer"),
                                    html.P(
                                        "Click the heatmap to choose (z*, Q*). The sliders stay in sync as precise fallbacks."
                                    ),
                                ],
                            ),
                            html.Div(
                                className="control-group",
                                children=[
                                    html.Label("LHAPDF set", htmlFor="set-dropdown"),
                                    dcc.Dropdown(
                                        id="set-dropdown",
                                        options=[{"label": name, "value": name} for name in SET_OPTIONS],
                                        value=INITIAL_SET,
                                        clearable=False,
                                        className="control-widget",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="control-group",
                                children=[
                                    html.Label("Member", htmlFor="member-input"),
                                    dcc.Input(
                                        id="member-input",
                                        type="number",
                                        min=0,
                                        step=1,
                                        value=DEFAULT_MEMBER,
                                        className="numeric-input",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="control-group",
                                children=[
                                    html.Label("PID", htmlFor="pid-dropdown"),
                                    dcc.Dropdown(id="pid-dropdown", clearable=False, className="control-widget"),
                                ],
                            ),
                            html.Div(
                                className="control-group",
                                children=[
                                    html.Label("Quantity"),
                                    dcc.RadioItems(
                                        id="quantity-radio",
                                        options=[
                                            {"label": "z * D(z, Q)", "value": "xfx"},
                                            {"label": "D(z, Q)", "value": "fx"},
                                        ],
                                        value="fx",
                                        className="stacked-radio",
                                        inputClassName="stacked-radio__input",
                                        labelClassName="stacked-radio__label",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="divider",
                            ),
                            html.Div(
                                className="control-group",
                                children=[
                                    html.Label("z scale"),
                                    dcc.RadioItems(
                                        id="z-scale-radio",
                                        options=[
                                            {"label": "log", "value": "log"},
                                            {"label": "linear", "value": "linear"},
                                        ],
                                        value="log",
                                        className="segmented-radio",
                                        inputClassName="segmented-radio__input",
                                        labelClassName="segmented-radio__label",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="control-group",
                                children=[
                                    html.Label("Q scale"),
                                    dcc.RadioItems(
                                        id="q-scale-radio",
                                        options=[
                                            {"label": "log", "value": "log"},
                                            {"label": "linear", "value": "linear"},
                                        ],
                                        value="log",
                                        className="segmented-radio",
                                        inputClassName="segmented-radio__input",
                                        labelClassName="segmented-radio__label",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="control-group",
                                children=[
                                    html.Label("Color"),
                                    dcc.RadioItems(
                                        id="color-scale-radio",
                                        options=[
                                            {"label": "linear", "value": "linear"},
                                            {"label": "log", "value": "log"},
                                            {"label": "symlog", "value": "symlog"},
                                        ],
                                        value="linear",
                                        className="stacked-radio",
                                        inputClassName="stacked-radio__input",
                                        labelClassName="stacked-radio__label",
                                    ),
                                ],
                            ),
                            html.Div(className="divider"),
                            html.Div(
                                className="control-group",
                                children=[
                                    html.Label("z*", htmlFor="z-star-input"),
                                    dcc.Input(id="z-star-input", type="number", className="numeric-input"),
                                    dcc.Slider(id="z-slider", tooltip={"placement": "bottom"}),
                                ],
                            ),
                            html.Div(
                                className="control-group",
                                children=[
                                    html.Label("Q*", htmlFor="q-star-input"),
                                    dcc.Input(id="q-star-input", type="number", className="numeric-input"),
                                    dcc.Slider(id="q-slider", tooltip={"placement": "bottom"}),
                                ],
                            ),
                            html.Details(
                                className="advanced-panel",
                                children=[
                                    html.Summary("Advanced"),
                                    html.Div(
                                        className="advanced-grid",
                                        children=[
                                            html.Div(
                                                className="control-group compact",
                                                children=[
                                                    html.Label("z min", htmlFor="z-min-input"),
                                                    dcc.Input(id="z-min-input", type="number", className="numeric-input"),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group compact",
                                                children=[
                                                    html.Label("z max", htmlFor="z-max-input"),
                                                    dcc.Input(id="z-max-input", type="number", className="numeric-input"),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group compact",
                                                children=[
                                                    html.Label("Q min", htmlFor="q-min-input"),
                                                    dcc.Input(id="q-min-input", type="number", className="numeric-input"),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group compact",
                                                children=[
                                                    html.Label("Q max", htmlFor="q-max-input"),
                                                    dcc.Input(id="q-max-input", type="number", className="numeric-input"),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group compact",
                                                children=[
                                                    html.Label("z samples", htmlFor="z-points-input"),
                                                    dcc.Input(
                                                        id="z-points-input",
                                                        type="number",
                                                        min=10,
                                                        step=1,
                                                        className="numeric-input",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group compact",
                                                children=[
                                                    html.Label("Q samples", htmlFor="q-points-input"),
                                                    dcc.Input(
                                                        id="q-points-input",
                                                        type="number",
                                                        min=10,
                                                        step=1,
                                                        className="numeric-input",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group compact",
                                                children=[
                                                    html.Label("Interpolation", htmlFor="interpolation-dropdown"),
                                                    dcc.Dropdown(
                                                        id="interpolation-dropdown",
                                                        options=[
                                                            {"label": "nearest", "value": "nearest"},
                                                            {"label": "smooth", "value": "smooth"},
                                                        ],
                                                        value="smooth",
                                                        clearable=False,
                                                        className="control-widget",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group compact",
                                                children=[
                                                    html.Label("Grid overlay"),
                                                    dcc.Checklist(
                                                        id="grid-overlay-checklist",
                                                        options=[{"label": "show cell grid", "value": "grid"}],
                                                        value=[],
                                                        className="single-checklist",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(id="dataset-summary", className="dataset-summary"),
                        ],
                    ),
                    html.Main(
                        className="plot-panel",
                        children=[
                            dcc.Graph(
                                id="heatmap-graph",
                                className="plot-card heatmap-card",
                                config={"displaylogo": False, "responsive": True},
                            ),
                            html.Div(
                                className="projection-row",
                                children=[
                                    dcc.Graph(
                                        id="z-projection-graph",
                                        className="plot-card projection-card",
                                        config={"displaylogo": False, "responsive": True},
                                    ),
                                    dcc.Graph(
                                        id="q-projection-graph",
                                        className="plot-card projection-card",
                                        config={"displaylogo": False, "responsive": True},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )

    @app.callback(
        Output("member-input", "max"),
        Output("member-input", "value"),
        Output("pid-dropdown", "options"),
        Output("pid-dropdown", "value"),
        Output("dataset-summary", "children"),
        Output("z-min-input", "value"),
        Output("z-max-input", "value"),
        Output("q-min-input", "value"),
        Output("q-max-input", "value"),
        Output("z-points-input", "value"),
        Output("q-points-input", "value"),
        Input("set-dropdown", "value"),
        Input("member-input", "value"),
        State("pid-dropdown", "value"),
    )
    def load_dataset(set_name: str, requested_member: Any, current_pid: int | None) -> tuple[Any, ...]:
        if not set_name:
            return 0, 0, [], None, "Choose a set to begin.", no_update, no_update, no_update, no_update, no_update, no_update

        meta = dataset_metadata(set_name, 0)
        max_member = max(meta["member_count"] - 1, 0)
        member = sanitize_member(requested_member, max_member)
        meta = dataset_metadata(set_name, member)
        pid_options = [{"label": format_pid(pid), "value": pid} for pid in meta["flavors"]]
        pid_value = preferred_pid(meta["flavors"], current_pid)
        description = meta["description"] or "No description available."
        summary = (
            f"{description} "
            f"Domain: z in [{format_number(meta['z_min'])}, {format_number(meta['z_max'])}], "
            f"Q in [{format_number(meta['q_min'])}, {format_number(meta['q_max'])}]."
        )
        return (
            max_member,
            member,
            pid_options,
            pid_value,
            summary,
            meta["z_min"],
            meta["z_max"],
            meta["q_min"],
            meta["q_max"],
            DEFAULT_Z_POINTS,
            DEFAULT_Q_POINTS,
        )

    @app.callback(
        Output("z-slider", "min"),
        Output("z-slider", "max"),
        Output("z-slider", "marks"),
        Output("z-slider", "step"),
        Output("q-slider", "min"),
        Output("q-slider", "max"),
        Output("q-slider", "marks"),
        Output("q-slider", "step"),
        Input("z-min-input", "value"),
        Input("z-max-input", "value"),
        Input("q-min-input", "value"),
        Input("q-max-input", "value"),
        Input("z-scale-radio", "value"),
        Input("q-scale-radio", "value"),
    )
    def configure_sliders(
        z_min_raw: Any,
        z_max_raw: Any,
        q_min_raw: Any,
        q_max_raw: Any,
        z_scale: str,
        q_scale: str,
    ) -> tuple[Any, ...]:
        z_min, z_max = sanitize_range(z_min_raw, z_max_raw, 1e-4, 1.0)
        q_min, q_max = sanitize_range(q_min_raw, q_max_raw, 1.0, 100.0)
        log_z = z_scale == "log"
        log_q = q_scale == "log"
        return (
            encode_slider(z_min, log_z),
            encode_slider(z_max, log_z),
            make_marks(z_min, z_max, log_z),
            slider_step(z_min, z_max, log_z),
            encode_slider(q_min, log_q),
            encode_slider(q_max, log_q),
            make_marks(q_min, q_max, log_q),
            slider_step(q_min, q_max, log_q),
        )

    @app.callback(
        Output("z-star-input", "value"),
        Output("q-star-input", "value"),
        Output("z-slider", "value"),
        Output("q-slider", "value"),
        Input("heatmap-graph", "clickData"),
        Input("z-star-input", "value"),
        Input("q-star-input", "value"),
        Input("z-slider", "value"),
        Input("q-slider", "value"),
        Input("z-min-input", "value"),
        Input("z-max-input", "value"),
        Input("q-min-input", "value"),
        Input("q-max-input", "value"),
        Input("z-scale-radio", "value"),
        Input("q-scale-radio", "value"),
    )
    def sync_selection(
        click_data: dict[str, Any] | None,
        z_input: Any,
        q_input: Any,
        z_slider: Any,
        q_slider: Any,
        z_min_raw: Any,
        z_max_raw: Any,
        q_min_raw: Any,
        q_max_raw: Any,
        z_scale: str,
        q_scale: str,
    ) -> tuple[float, float, float, float]:
        z_min, z_max = sanitize_range(z_min_raw, z_max_raw, 1e-4, 1.0)
        q_min, q_max = sanitize_range(q_min_raw, q_max_raw, 1.0, 100.0)
        log_z = z_scale == "log"
        log_q = q_scale == "log"
        fallback_z = midpoint(z_min, z_max, log_z)
        fallback_q = midpoint(q_min, q_max, log_q)
        current_z = clamp(sanitize_positive(z_input, fallback_z), z_min, z_max)
        current_q = clamp(sanitize_positive(q_input, fallback_q), q_min, q_max)
        triggered = callback_context.triggered_id

        if triggered == "heatmap-graph" and click_data and click_data.get("points"):
            point = click_data["points"][0]
            current_z = clamp(float(point["x"]), z_min, z_max)
            current_q = clamp(float(point["y"]), q_min, q_max)
        elif triggered == "z-slider":
            current_z = clamp(decode_slider(z_slider, log_z, fallback_z), z_min, z_max)
        elif triggered == "q-slider":
            current_q = clamp(decode_slider(q_slider, log_q, fallback_q), q_min, q_max)
        elif triggered == "z-star-input":
            current_z = clamp(sanitize_positive(z_input, fallback_z), z_min, z_max)
        elif triggered == "q-star-input":
            current_q = clamp(sanitize_positive(q_input, fallback_q), q_min, q_max)

        return (
            current_z,
            current_q,
            encode_slider(current_z, log_z),
            encode_slider(current_q, log_q),
        )

    @app.callback(
        Output("heatmap-graph", "figure"),
        Output("z-projection-graph", "figure"),
        Output("q-projection-graph", "figure"),
        Input("set-dropdown", "value"),
        Input("member-input", "value"),
        Input("pid-dropdown", "value"),
        Input("quantity-radio", "value"),
        Input("z-scale-radio", "value"),
        Input("q-scale-radio", "value"),
        Input("color-scale-radio", "value"),
        Input("z-star-input", "value"),
        Input("q-star-input", "value"),
        Input("z-min-input", "value"),
        Input("z-max-input", "value"),
        Input("q-min-input", "value"),
        Input("q-max-input", "value"),
        Input("z-points-input", "value"),
        Input("q-points-input", "value"),
        Input("interpolation-dropdown", "value"),
        Input("grid-overlay-checklist", "value"),
    )
    def render_plots(
        set_name: str,
        member_raw: Any,
        pid: int | None,
        quantity: str,
        z_scale: str,
        q_scale: str,
        color_mode: str,
        z_star_raw: Any,
        q_star_raw: Any,
        z_min_raw: Any,
        z_max_raw: Any,
        q_min_raw: Any,
        q_max_raw: Any,
        z_points_raw: Any,
        q_points_raw: Any,
        interpolation: str,
        grid_overlay: list[str] | None,
    ) -> tuple[go.Figure, go.Figure, go.Figure]:
        if not set_name or pid is None:
            empty = empty_figure("Choose a set and PID to begin.")
            return empty, empty, empty

        member_meta = dataset_metadata(set_name, 0)
        member = sanitize_member(member_raw, max(member_meta["member_count"] - 1, 0))
        meta = dataset_metadata(set_name, member)
        z_min, z_max = sanitize_range(z_min_raw, z_max_raw, meta["z_min"], meta["z_max"])
        q_min, q_max = sanitize_range(q_min_raw, q_max_raw, meta["q_min"], meta["q_max"])
        z_points = sanitize_int(z_points_raw, DEFAULT_Z_POINTS)
        q_points = sanitize_int(q_points_raw, DEFAULT_Q_POINTS)
        log_z = z_scale == "log"
        log_q = q_scale == "log"
        z_star = clamp(sanitize_positive(z_star_raw, midpoint(z_min, z_max, log_z)), z_min, z_max)
        q_star = clamp(sanitize_positive(q_star_raw, midpoint(q_min, q_max, log_q)), q_min, q_max)
        interpolation_mode = "smooth" if interpolation == "smooth" else "nearest"
        show_cell_grid = bool(grid_overlay and "grid" in grid_overlay)

        heatmap = heatmap_figure(
            set_name,
            member,
            pid,
            quantity,
            color_mode,
            z_min,
            z_max,
            q_min,
            q_max,
            z_points,
            q_points,
            log_z,
            log_q,
            z_star,
            q_star,
            interpolation_mode,
            show_cell_grid,
        )

        z_axis, z_values = sample_z_projection(
            set_name,
            member,
            pid,
            quantity,
            z_min,
            z_max,
            q_star,
            z_points,
            log_z,
        )
        q_axis, q_values = sample_q_projection(
            set_name,
            member,
            pid,
            quantity,
            q_min,
            q_max,
            z_star,
            q_points,
            log_q,
        )
        selected_value = evaluate_quantity(get_pdf(set_name, member), pid, z_star, q_star, quantity)
        y_title = quantity_label(quantity)
        z_projection = projection_figure(
            z_axis,
            z_values,
            z_star,
            selected_value,
            f"{y_title} vs z<br><sup>Q* = {format_number(q_star)}</sup>",
            "z",
            y_title,
            log_z,
            LINE_A,
        )
        q_projection = projection_figure(
            q_axis,
            q_values,
            q_star,
            selected_value,
            f"{y_title} vs Q<br><sup>z* = {format_number(z_star)}</sup>",
            "Q",
            y_title,
            log_q,
            LINE_B,
        )
        return heatmap, z_projection, q_projection

    return app


def main() -> None:
    app = build_app()
    app.run(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    main()
