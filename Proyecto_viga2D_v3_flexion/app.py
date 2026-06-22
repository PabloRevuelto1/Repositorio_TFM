"""
app.py
======
Dashboard interactivo (Dash + Plotly) de la elastodinámica de una viga empotrada
2D, en su versión SIMPLIFICADA: un único escenario fijo (config.VIGA) resuelto por
DOS métodos que se comparan objetivamente:

  * PINNsFormer  — red neuronal informada por la física (la solución aprendida).
  * Elementos Finitos (scikit-fem) — solución de REFERENCIA (*ground truth*).

La comparación es a la vez:
  * GRÁFICA  — malla deformada (conmutable PINN/FEM), oscilación del extremo y
    perfil de σ_xx superpuestos.
  * MÉTRICA  — concordancia de campos (error L2) y tabla de "ganador por
    categoría" (precisión, tiempo de cómputo, ubicación de la tensión máxima…).

Inferencia/solución instantáneas: NO reentrena. Carga checkpoints/best_model.pth
(o lo descarga de W&B) y resuelve el FEM en el momento.

Ejecución:
    python app.py            # http://127.0.0.1:8050
"""

import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import torch
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

import config as C
from utils.inference import load_model, evaluate_fields
from utils.registry import download_checkpoint
from utils.metrics import (simulation_metrics, sanity_checks, compare_with_reference,
                           validation_residual, compare_fields_pinn_fem, winner_table)
from fem.fem_solver import solve_fem

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAT = C.MATERIAL                     # único material (acero)


def _try_load(path):
    """Intenta cargar un checkpoint; devuelve (model, ckpt) o (None, None)."""
    try:
        model, ckpt = load_model(path, device=DEVICE)
        print(f"[app] modelo cargado desde {path} (best_loss={ckpt.get('best_loss', 'NA')})")
        return model, ckpt
    except Exception as exc:
        print(f"[app] no se pudo cargar '{path}': {exc}")
        return None, None


def _get_model():
    """Devuelve (modelo, checkpoint_dict). Ambos None si no se pudo cargar.

    Estrategia robusta: primero el checkpoint LOCAL; si no existe o es de una
    versión INCOMPATIBLE (p. ej. un modelo antiguo de 7 entradas), se descarga el
    último de W&B y se carga ese. Así un checkpoint local viejo no "esconde" al
    modelo bueno publicado en W&B.
    """
    path = C.ENTRENAMIENTO.checkpoint
    if os.path.exists(path):
        model, ckpt = _try_load(path)
        if model is not None:
            return model, ckpt
        print("[app] el checkpoint local no es válido (¿versión antigua?); probando W&B...")
    else:
        print(f"[app] checkpoint local '{path}' no encontrado; probando W&B...")

    if not os.getenv("WANDB_API_KEY"):
        print("[app] ⚠️  WANDB_API_KEY no definida en .env: no puedo descargar de W&B.")
    downloaded = download_checkpoint()
    if downloaded:
        model, ckpt = _try_load(downloaded)
        if model is not None:
            return model, ckpt
    print("[app] AVISO: no hay modelo disponible. Reentrena con el código actual.")
    return None, None


MODEL, CKPT = _get_model()

# Cachés perezosas: el FEM y la última simulación se reutilizan entre clics.
FEM_DATA = None
SIM = {}        # {"pinn": dict, "fem": dict, "scale": float, "L": ..., "c": ...}


def _fem_data():
    global FEM_DATA
    if FEM_DATA is None:
        print("[app] resolviendo FEM de referencia (scikit-fem)...")
        FEM_DATA = solve_fem(MAT)
    return FEM_DATA


# ----------------------------------------------------------------------
# Estilos (tema oscuro)
# ----------------------------------------------------------------------
BG, PANEL, ACCENT, TEXT, MUTED = "#0e1117", "#161b22", "#00d4ff", "#e6edf3", "#8b949e"
FEM_COL = "#ffb454"
OK_COL, WARN_COL, BAD_COL = "#3fb950", "#d29922", "#f85149"
TONE = {"ok": OK_COL, "warn": WARN_COL, "bad": BAD_COL}
CARD = {"backgroundColor": PANEL, "borderRadius": "12px", "padding": "18px",
        "marginBottom": "16px", "border": "1px solid #21262d"}
LABEL = {"color": TEXT, "fontWeight": 600, "fontSize": "14px", "marginTop": "12px"}
TD = {"padding": "4px 8px", "fontSize": "13px", "color": TEXT}
TD_R = {**TD, "textAlign": "right", "fontFamily": "monospace", "color": ACCENT}

app = Dash(__name__, title="Viga 2D · PINNsFormer vs FEM")
server = app.server


# ----------------------------------------------------------------------
# Componentes de tablas
# ----------------------------------------------------------------------
def _rows_table(rows):
    body = [html.Tr([html.Td(lbl, style=TD), html.Td(val, style=TD_R)]) for lbl, val in rows]
    return html.Table(html.Tbody(body), style={"width": "100%", "borderCollapse": "collapse"})


def _checks_table(rows):
    body = []
    for lbl, val, ok in rows:
        icon, col = ("·", MUTED) if ok is None else (("✓", OK_COL) if ok else ("✗", BAD_COL))
        body.append(html.Tr([
            html.Td(icon, style={**TD, "color": col, "fontWeight": 700, "width": "22px"}),
            html.Td(lbl, style=TD), html.Td(val, style=TD_R)]))
    return html.Table(html.Tbody(body), style={"width": "100%", "borderCollapse": "collapse"})


def _winner_table(rows):
    """Tabla de 4 columnas (categoría, PINN, FEM, ganador) con el ganador coloreado."""
    head = html.Tr([html.Th(h, style={**TD, "color": MUTED, "textAlign": "left",
                                       "borderBottom": f"1px solid #30363d"})
                    for h in ("Categoría", "PINN", "FEM", "Ganador")])
    body = []
    for cat, pv, fv, win in rows:
        wcol = {"PINN": ACCENT, "FEM": FEM_COL}.get(win, MUTED)
        body.append(html.Tr([
            html.Td(cat, style=TD),
            html.Td(pv, style={**TD, "fontFamily": "monospace", "color": ACCENT}),
            html.Td(fv, style={**TD, "fontFamily": "monospace", "color": FEM_COL}),
            html.Td(win, style={**TD, "fontWeight": 700, "color": wcol})]))
    return html.Table([html.Thead(head), html.Tbody(body)],
                      style={"width": "100%", "borderCollapse": "collapse"})


def _quality_badge(comps):
    if not comps:
        return html.Span("sin métricas", style={"color": MUTED})
    val = C.VALIDACION
    worst = max(comps.get("res", 0), comps.get("load", 0), comps.get("bc", 0),
                comps.get("ic", 0), comps.get("eq", 0))
    tone, txt = ("ok", "BUENO") if worst < val.umbral_loss_ok else \
                (("warn", "ACEPTABLE") if worst < val.umbral_loss_warn else ("bad", "REVISAR"))
    return html.Span(f"●  {txt}", style={"color": TONE[tone], "fontWeight": 700})


def _convergence_figure(history):
    fig = go.Figure()
    if history:
        steps = [h["step"] for h in history]
        for key, color, name in (("loss", ACCENT, "total"), ("res", "#a371f7", "L_res"),
                                  ("load", "#58a6ff", "L_load"), ("bc", "#3fb950", "L_bc"),
                                  ("ic", "#d29922", "L_ic"), ("eq", "#f778ba", "L_eq")):
            ys = [max(h.get(key, float("nan")), 1e-12) for h in history]
            fig.add_trace(go.Scatter(x=steps, y=ys, mode="lines", name=name,
                                     line=dict(color=color, width=2)))
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      margin=dict(l=50, r=10, t=10, b=35), height=200,
                      yaxis=dict(type="log", title="pérdida"), xaxis_title="época",
                      legend=dict(orientation="h", y=1.15, font=dict(size=10)))
    return fig


def _training_card(ckpt):
    if not ckpt:
        return html.Div(style=CARD, children=[
            html.H4("📊 Métricas de entrenamiento", style={"color": TEXT, "marginTop": 0}),
            html.Div("Sin checkpoint cargado (entrena primero, ver README).",
                     style={"color": MUTED, "fontSize": "13px"})])
    comps = ckpt.get("final_comps", {})
    rows = [
        ("Pérdida total (mejor)", f"{ckpt.get('best_loss', float('nan')):.3e}"),
        ("L_res  (residuo EDP)", f"{comps.get('res', float('nan')):.3e}"),
        ("L_load (carga ξ=1)", f"{comps.get('load', float('nan')):.3e}"),
        ("L_bc   (contorno homog.)", f"{comps.get('bc', float('nan')):.3e}"),
        ("L_ic   (inicial)", f"{comps.get('ic', float('nan')):.3e}"),
        ("L_eq   (equilibrio seccional)", f"{comps.get('eq', float('nan')):.3e}"),
        ("Parámetros del modelo", f"{ckpt.get('n_params', 0):,}"),
        ("Épocas Adam / L-BFGS", f"{ckpt.get('adam_epochs', '—')} / {ckpt.get('lbfgs_epochs', '—')}"),
    ]
    return html.Div(style=CARD, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                 children=[html.H4("📊 Métricas de entrenamiento", style={"color": TEXT, "margin": 0}),
                           _quality_badge(comps)]),
        _rows_table(rows),
        html.Div("Convergencia (escala log)", style={"color": MUTED, "fontSize": "12px", "marginTop": "10px"}),
        dcc.Graph(figure=_convergence_figure(ckpt.get("history", [])), config={"displayModeBar": False}),
    ])


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
def _scenario_rows():
    v = C.VIGA
    return _rows_table([
        ("Material", MAT.nombre),
        ("Longitud L", f"{v.L:.2f} m"),
        ("Semi-altura c", f"{v.c:.3f} m"),
        ("Carga máx. P_max", f"{v.P_max / 1e3:.0f} kN/m"),
        ("Rampa de carga t̂_ramp", f"{v.t_ramp_hat:.1f} (adim.)"),
        ("Horizonte temporal T̂", f"{C.DOMINIO.T_hat:.1f} (adim.)"),
    ])


controls = html.Div(style=CARD, children=[
    html.H3("⚙️  Escenario fijo", style={"color": ACCENT, "marginTop": 0}),
    html.Div("Geometría, carga y material son CONSTANTES (proyecto simplificado).",
             style={"color": MUTED, "fontSize": "12px", "marginBottom": "8px"}),
    _scenario_rows(),
    html.Button("▶  Simular y comparar (PINN vs FEM)", id="simular", n_clicks=0, style={
        "width": "100%", "marginTop": "18px", "padding": "14px", "backgroundColor": ACCENT,
        "color": "#001018", "border": "none", "borderRadius": "10px", "fontWeight": 700,
        "fontSize": "15px", "cursor": "pointer"}),
    html.Div(id="estado", style={"color": MUTED, "fontSize": "12px", "marginTop": "12px"}),
])

viz = html.Div(children=[
    html.Div(style=CARD, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                 children=[
            html.H4("Malla deformada · Von Mises [MPa]", style={"color": TEXT, "margin": 0}),
            dcc.RadioItems(id="fuente", value="PINN",
                           options=[{"label": " PINN", "value": "PINN"}, {"label": " FEM", "value": "FEM"}],
                           inline=True, labelStyle={"color": TEXT, "marginLeft": "12px"})]),
        dcc.Loading(dcc.Graph(id="main-graph", style={"height": "440px"}), color=ACCENT),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
        html.Div(style=CARD, children=[
            html.H4("Oscilación del extremo  v(L,0,t)", style={"color": TEXT, "marginTop": 0}),
            dcc.Graph(id="tip-graph", style={"height": "280px"})]),
        html.Div(style=CARD, children=[
            html.H4("σ_xx a lo largo de la viga (y=0)", style={"color": TEXT, "marginTop": 0}),
            dcc.Graph(id="sxx-graph", style={"height": "240px"}),
            html.Div("Instante de visualización", style={"color": MUTED, "fontSize": "12px"}),
            dcc.Slider(id="time-slider", min=0, max=1, value=0, step=1,
                       tooltip={"placement": "bottom", "always_visible": False})]),
    ]),
    html.Div(style=CARD, children=[
        html.H4("🏆 PINN vs FEM (ground truth)", style={"color": TEXT, "marginTop": 0}),
        html.Div(id="compare-metrics", children=html.Div(
            "Pulsa «Simular y comparar» para enfrentar ambos métodos.",
            style={"color": MUTED, "fontSize": "13px"}))]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
        html.Div(style=CARD, children=[
            html.H4("📐 Métricas de simulación (PINN)", style={"color": TEXT, "marginTop": 0}),
            html.Div(id="sim-metrics", children=html.Div(
                "Pulsa «Simular» para calcular.", style={"color": MUTED, "fontSize": "13px"}))]),
        html.Div(style=CARD, children=[
            html.H4("🔬 Validación física", style={"color": TEXT, "marginTop": 0}),
            html.Div(id="valid-metrics", children=html.Div(
                "Pulsa «Simular» para validar.", style={"color": MUTED, "fontSize": "13px"}))]),
    ]),
])

app.layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh", "padding": "24px",
                             "fontFamily": "Inter, system-ui, sans-serif"}, children=[
    html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px", "marginBottom": "20px"},
             children=[html.H1("🔧 Elastodinámica de Viga 2D", style={"color": TEXT, "margin": 0}),
                       html.Span("PINNsFormer vs Elementos Finitos · Tensión Plana",
                                 style={"color": ACCENT, "fontSize": "14px", "fontWeight": 600})]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "340px 1fr", "gap": "20px"},
             children=[html.Div([controls, _training_card(CKPT)]), viz]),
    dcc.Store(id="store"),
])


# ----------------------------------------------------------------------
# Callback principal: simulación, FEM y comparación
# ----------------------------------------------------------------------
@app.callback(
    Output("main-graph", "figure"), Output("tip-graph", "figure"),
    Output("store", "data"), Output("time-slider", "max"), Output("time-slider", "value"),
    Output("estado", "children"), Output("sim-metrics", "children"),
    Output("valid-metrics", "children"), Output("compare-metrics", "children"),
    Input("simular", "n_clicks"), State("fuente", "value"),
    prevent_initial_call=False,
)
def simular(_, fuente):
    if MODEL is None:
        warn = html.Div("Sin modelo.", style={"color": MUTED})
        return _empty_fig("Sin modelo"), _empty_fig(""), None, 1, 0, \
            "⚠️ Modelo no disponible (entrena y sube el checkpoint).", warn, warn, warn

    L, c = C.VIGA.L, C.VIGA.c

    t0 = time.time()
    pinn = evaluate_fields(MODEL, MAT, device=DEVICE)
    t_pinn = time.time() - t0

    t0 = time.time()
    fem = _fem_data()
    t_fem = time.time() - t0

    # Escala de exageración común (mismo factor para comparar de forma justa).
    max_disp = max(np.abs(pinn["u"]).max(), np.abs(pinn["v"]).max(),
                   np.abs(fem["u"]).max(), np.abs(fem["v"]).max(), 1e-12)
    scale = 0.18 * L / max_disp
    SIM.update(pinn=pinn, fem=fem, scale=scale, L=L, c=c)

    main_fig = _build_main_figure(SIM, fuente)
    tip_fig = _build_tip_figure(pinn, fem)

    sim = simulation_metrics(pinn, MAT)
    sim["sanity"] = sanity_checks(pinn)
    sim_card = _sim_metrics_children(sim)
    valid_card = _validation_children(sim)
    compare_card = _compare_children(pinn, fem, t_pinn, t_fem)

    store = {
        "x_line": (pinn["xi"] * L).tolist(),
        "sxx_pinn": (pinn["sxx"][:, pinn["j0"], :] / 1e6).tolist(),
        "sxx_fem": (fem["sxx"][:, fem["j0"], :] / 1e6).tolist(),
        "t_phys_ms": (pinn["t_phys"] * 1e3).tolist(),
    }
    nt = len(pinn["t_hat"])
    estado = (f"✅ PINN {t_pinn * 1e3:.0f} ms · FEM {t_fem * 1e3:.0f} ms · "
              f"{MAT.nombre} · deformada ×{scale:,.0f}")
    return main_fig, tip_fig, store, nt - 1, 0, estado, sim_card, valid_card, compare_card


@app.callback(Output("main-graph", "figure", allow_duplicate=True),
              Input("fuente", "value"), prevent_initial_call=True)
def cambiar_fuente(fuente):
    if not SIM:
        return _empty_fig("Pulsa «Simular y comparar»")
    return _build_main_figure(SIM, fuente)


def _sim_metrics_children(sim):
    safety = sim["safety"]
    verdict = html.Div(safety["veredicto"], style={
        "color": TONE[safety["tone"]], "fontWeight": 700, "fontSize": "14px", "margin": "4px 0 10px"})
    return [verdict, _rows_table(sim["rows"])]


def _compare_children(pinn, fem, t_pinn, t_fem):
    """Tarjeta de comparación: tabla de ganadores + concordancia de campos."""
    blocks = [
        html.Div("El FEM es la verdad de referencia; la tabla resume ventajas de cada método.",
                 style={"color": MUTED, "fontSize": "12px", "marginBottom": "6px"}),
        _winner_table(winner_table(pinn, fem, MAT, t_pinn, t_fem)),
        html.Div("Concordancia de campos (PINN medida contra FEM)",
                 style={"color": MUTED, "fontSize": "12px", "marginTop": "12px"}),
        _checks_table(compare_fields_pinn_fem(pinn, fem)),
    ]
    return blocks


def _validation_children(sim):
    def _sub(txt):
        return html.Div(txt, style={"color": MUTED, "fontSize": "12px", "marginTop": "10px"})
    blocks = []
    if "sanity" in sim:
        blocks.append(_sub("Nivel 2 · Comprobaciones físicas"))
        blocks.append(_checks_table(sim["sanity"]))
    if C.VALIDACION.referencia_analitica:
        blocks.append(_sub("Nivel 3 · Contraste Euler-Bernoulli"))
        blocks.append(_checks_table(compare_with_reference(sim, MAT, C.VIGA.L, C.VIGA.c, C.VIGA.P_max)))
    if C.VALIDACION.residuo_en_malla:
        blocks.append(_sub("Nivel 4 · Residuo EDP en malla nueva"))
        try:
            res = validation_residual(MODEL, MAT, C.DOMINIO, C.ENTRENAMIENTO,
                                      n=C.VALIDACION.n_residuo_check, device=DEVICE)
            blocks.append(_checks_table([("Residuo medio |r|²", f"{res:.2e}",
                                          res < C.VALIDACION.umbral_loss_warn)]))
        except Exception as exc:
            blocks.append(html.Div(f"Residuo no disponible ({exc}).",
                                   style={"color": MUTED, "fontSize": "12px"}))
    return blocks


@app.callback(Output("sxx-graph", "figure"),
              Input("time-slider", "value"), Input("store", "data"))
def actualizar_sxx(idx, store):
    if not store:
        return _empty_fig("")
    idx = int(idx or 0)
    x = store["x_line"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=store["sxx_pinn"][idx], mode="lines", name="PINN",
                             line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=x, y=store["sxx_fem"][idx], mode="lines", name="FEM",
                             line=dict(color=FEM_COL, width=2, dash="dash")))
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      margin=dict(l=50, r=20, t=30, b=40),
                      title=dict(text=f"t = {store['t_phys_ms'][idx]:.3f} ms",
                                 font=dict(size=12, color=MUTED)),
                      xaxis_title="x [m]", yaxis_title="σ_xx [MPa]",
                      legend=dict(orientation="h", y=1.2, font=dict(size=10)))
    return fig


# ----------------------------------------------------------------------
# Constructores de figuras
# ----------------------------------------------------------------------
def _empty_fig(text):
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      annotations=[dict(text=text, showarrow=False, font=dict(color=MUTED, size=14))])
    return fig


def _build_main_figure(sim, fuente):
    """Malla deformada animada coloreada por Von Mises (MPa) del método elegido."""
    data = sim["fem"] if fuente == "FEM" else sim["pinn"]
    L, c, scale = sim["L"], sim["c"], sim["scale"]
    x, y, u, v = data["x"], data["y"], data["u"], data["v"]
    von = data["von_mises"] / 1e6
    nt = x.shape[0]
    # Escala de color común a ambos métodos para comparar de forma justa.
    vmax = max(sim["pinn"]["von_mises"].max(), sim["fem"]["von_mises"].max(), 1e-9) / 1e6

    def frame_trace(i):
        xd = (x[i] + scale * u[i]).ravel()
        yd = (y[i] + scale * v[i]).ravel()
        return go.Scatter(x=xd, y=yd, mode="markers",
                          marker=dict(size=7, color=von[i].ravel(), colorscale="Turbo",
                                      cmin=0, cmax=vmax, colorbar=dict(title="MPa"), symbol="square"),
                          hovertemplate="x=%{x:.3f} m<br>y=%{y:.4f} m<extra></extra>")

    frames = [go.Frame(data=[frame_trace(i)], name=str(i)) for i in range(nt)]
    fig = go.Figure(data=[frame_trace(0)], frames=frames)
    play = dict(label="▶ Play", method="animate",
                args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)])
    pause = dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
    steps = [dict(method="animate", label=f"{data['t_phys'][i] * 1e3:.2f}",
                  args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
             for i in range(nt)]
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        margin=dict(l=40, r=20, t=20, b=20),
        title=dict(text=fuente, font=dict(color=ACCENT if fuente == "PINN" else FEM_COL, size=13)),
        xaxis=dict(title="x [m]", range=[-0.05 * L, 1.15 * L], zeroline=False),
        yaxis=dict(title="y [m]", scaleanchor="x", scaleratio=1.0, zeroline=False),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.12, direction="left",
                          buttons=[play, pause])],
        sliders=[dict(active=0, x=0.12, len=0.85, y=0, currentvalue=dict(prefix="t = ", suffix=" ms"),
                      steps=steps)])
    return fig


def _build_tip_figure(pinn, fem):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pinn["t_phys"] * 1e3, y=pinn["tip_v"] * 1e6, mode="lines",
                             name="PINN", line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=fem["t_phys"] * 1e3, y=fem["tip_v"] * 1e6, mode="lines",
                             name="FEM", line=dict(color=FEM_COL, width=2, dash="dash")))
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      margin=dict(l=55, r=20, t=20, b=40), xaxis_title="t [ms]",
                      yaxis_title="v(L,0,t) [µm]",
                      legend=dict(orientation="h", y=1.18, font=dict(size=10)))
    return fig


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    app.run(debug=False, host="0.0.0.0", port=port)
