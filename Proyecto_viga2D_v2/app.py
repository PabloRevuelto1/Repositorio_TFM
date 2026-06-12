"""
app.py
======
Dashboard interactivo (Dash + Plotly) para visualizar la elastodinámica de
una viga empotrada 2D resuelta con PINNsFormer.

Características:
  * Tema oscuro industrial/ingenieril, limpio y profesional.
  * Panel lateral de control: geometría (L, c), carga (P_max, t_impacto),
    selector de material (Acero / Aluminio) y botón "Simular Dinámica".
  * Visualización central:
      - Mapa animado de la malla DEFORMADA coloreada por tensión de Von Mises
        (Play/Pause/slider de tiempo nativos de Plotly).
      - Gráfica 1: desplazamiento del extremo libre v(L,0,t) en el tiempo.
      - Gráfica 2: distribución de σ_xx a lo largo del eje X en el instante
        seleccionado (slider).
  * Inferencia instantánea: NO reentrena. Carga checkpoints/best_model.pth
    (o lo descarga de W&B) y evalúa la red para los parámetros de los sliders.

Ejecución:
    python app.py            # http://127.0.0.1:8050
"""

import os

# Carga las claves del .env (WANDB_API_KEY, etc.) para poder descargar el
# modelo desde W&B al ejecutar la app en local.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import torch
from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go

import config as C
from utils.inference import load_model, evaluate_fields
from utils.registry import download_checkpoint
from utils.metrics import (simulation_metrics, sanity_checks,
                           compare_with_reference, validation_residual)

# ----------------------------------------------------------------------
# Carga del modelo (local → W&B → None)
# ----------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_model():
    """Devuelve (modelo, checkpoint_dict). Ambos None si no se pudo cargar."""
    path = C.ENTRENAMIENTO.checkpoint
    if not os.path.exists(path):
        print(f"[app] checkpoint local '{path}' no encontrado; intentando descargar de W&B...")
        if not os.getenv("WANDB_API_KEY"):
            print("[app] ⚠️  WANDB_API_KEY no está definida. Crea/edita el archivo .env "
                  "(cópialo de .env.example) con tu clave de W&B y reinicia la app.")
        downloaded = download_checkpoint()
        if downloaded:
            print(f"[app] modelo descargado de W&B a '{downloaded}'.")
            path = downloaded
    try:
        model, ckpt = load_model(path, device=DEVICE)
        print(f"[app] modelo cargado desde {path} (best_loss={ckpt.get('best_loss', 'NA')})")
        return model, ckpt
    except Exception as exc:
        print(f"[app] AVISO: no se pudo cargar el modelo ({exc}).")
        print("[app] La interfaz se mostrará, pero hay que entrenar primero (ver README).")
        return None, None


MODEL, CKPT = _get_model()

# ----------------------------------------------------------------------
# Estilos (tema oscuro)
# ----------------------------------------------------------------------
BG = "#0e1117"
PANEL = "#161b22"
ACCENT = "#00d4ff"
TEXT = "#e6edf3"
MUTED = "#8b949e"

CARD = {"backgroundColor": PANEL, "borderRadius": "12px", "padding": "18px",
        "marginBottom": "16px", "border": "1px solid #21262d"}
LABEL = {"color": TEXT, "fontWeight": 600, "fontSize": "14px", "marginTop": "12px"}

app = Dash(__name__, title="Viga 2D · PINNsFormer")
server = app.server  # para despliegue WSGI (gunicorn) si se desea


def slider(id_, mn, mx, val, step, marks_unit=""):
    marks = {float(v): {"label": f"{v:g}{marks_unit}", "style": {"color": MUTED, "fontSize": "10px"}}
             for v in np.linspace(mn, mx, 5)}
    return dcc.Slider(id=id_, min=mn, max=mx, value=val, step=step, marks=marks,
                      tooltip={"placement": "bottom", "always_visible": True})


# ----------------------------------------------------------------------
# Componentes de presentación de métricas (docs/03 §12 y §13)
# ----------------------------------------------------------------------
OK_COL, WARN_COL, BAD_COL = "#3fb950", "#d29922", "#f85149"
TONE = {"ok": OK_COL, "warn": WARN_COL, "bad": BAD_COL}

TD = {"padding": "4px 8px", "fontSize": "13px", "color": TEXT}
TD_R = {**TD, "textAlign": "right", "fontFamily": "monospace", "color": ACCENT}


def _rows_table(rows):
    """Tabla simple de (etiqueta, valor) para métricas escalares."""
    body = [html.Tr([html.Td(lbl, style=TD), html.Td(val, style=TD_R)]) for lbl, val in rows]
    return html.Table(html.Tbody(body), style={"width": "100%", "borderCollapse": "collapse"})


def _checks_table(rows):
    """Tabla de (etiqueta, valor, ok) con icono ✓/✗ coloreado. ok=None → neutro."""
    body = []
    for lbl, val, ok in rows:
        if ok is None:
            icon, col = "·", MUTED
        else:
            icon, col = ("✓", OK_COL) if ok else ("✗", BAD_COL)
        body.append(html.Tr([
            html.Td(icon, style={**TD, "color": col, "fontWeight": 700, "width": "22px"}),
            html.Td(lbl, style=TD),
            html.Td(val, style=TD_R),
        ]))
    return html.Table(html.Tbody(body), style={"width": "100%", "borderCollapse": "collapse"})


def _quality_badge(comps, weights):
    """Semáforo de calidad del modelo según las pérdidas finales (§12.1)."""
    if not comps:
        return html.Span("sin métricas", style={"color": MUTED})
    val = C.VALIDACION
    worst = max(comps.get("res", 0), comps.get("load", 0),
                comps.get("bc", 0), comps.get("ic", 0))
    if worst < val.umbral_loss_ok:
        tone, txt = "ok", "BUENO"
    elif worst < val.umbral_loss_warn:
        tone, txt = "warn", "ACEPTABLE"
    else:
        tone, txt = "bad", "REVISAR"
    return html.Span(f"●  {txt}", style={"color": TONE[tone], "fontWeight": 700})


def _convergence_figure(history):
    """Curva de convergencia (pérdida total y componentes vs. época, escala log)."""
    fig = go.Figure()
    if history:
        steps = [h["step"] for h in history]
        for key, color, name in (("loss", ACCENT, "total"), ("res", "#a371f7", "L_res"),
                                  ("load", "#58a6ff", "L_load"),
                                  ("bc", "#3fb950", "L_bc"), ("ic", "#d29922", "L_ic")):
            ys = [max(h.get(key, float("nan")), 1e-12) for h in history]
            fig.add_trace(go.Scatter(x=steps, y=ys, mode="lines", name=name,
                                     line=dict(color=color, width=2)))
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      margin=dict(l=50, r=10, t=10, b=35), height=200,
                      yaxis=dict(type="log", title="pérdida"), xaxis_title="época",
                      legend=dict(orientation="h", y=1.15, font=dict(size=10)))
    return fig


def _training_card(ckpt):
    """Tarjeta estática con las métricas de entrenamiento del checkpoint (§13.1)."""
    if not ckpt:
        return html.Div(style=CARD, children=[
            html.H4("📊 Métricas de entrenamiento", style={"color": TEXT, "marginTop": 0}),
            html.Div("Sin checkpoint cargado (entrena primero, ver README).",
                     style={"color": MUTED, "fontSize": "13px"})])

    comps = ckpt.get("final_comps", {})
    weights = ckpt.get("loss_weights", {})
    rows = [
        ("Pérdida total (mejor)", f"{ckpt.get('best_loss', float('nan')):.3e}"),
        ("L_res  (residuo EDP)", f"{comps.get('res', float('nan')):.3e}"),
        ("L_load (carga ξ=1)", f"{comps.get('load', float('nan')):.3e}"),
        ("L_bc   (contorno homog.)", f"{comps.get('bc', float('nan')):.3e}"),
        ("L_ic   (inicial)", f"{comps.get('ic', float('nan')):.3e}"),
        ("Material de entrenamiento", str(ckpt.get("material_entrenamiento", "—"))),
        ("Parámetros del modelo", f"{ckpt.get('n_params', 0):,}"),
        ("Épocas Adam / L-BFGS", f"{ckpt.get('adam_epochs', '—')} / {ckpt.get('lbfgs_epochs', '—')}"),
    ]
    return html.Div(style=CARD, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "alignItems": "center"}, children=[
            html.H4("📊 Métricas de entrenamiento", style={"color": TEXT, "margin": 0}),
            _quality_badge(comps, weights),
        ]),
        _rows_table(rows),
        html.Div("Convergencia (escala log)", style={"color": MUTED, "fontSize": "12px",
                                                     "marginTop": "10px"}),
        dcc.Graph(figure=_convergence_figure(ckpt.get("history", [])),
                  config={"displayModeBar": False}),
    ])


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
controls = html.Div(style=CARD, children=[
    html.H3("⚙️  Parámetros", style={"color": ACCENT, "marginTop": 0}),

    html.Div("Material", style=LABEL),
    dcc.Dropdown(id="material", value="Acero", clearable=False,
                 options=[{"label": f"{m.nombre}  (E={m.E/1e9:.0f} GPa, ν={m.nu})", "value": k}
                          for k, m in C.MATERIALES.items()],
                 style={"color": "#000"}),

    html.Div("Longitud L  [m]", style=LABEL),
    slider("L", C.DOMINIO.L_hat[0] * C.L_REF, C.DOMINIO.L_hat[1] * C.L_REF, 1.5, 0.05, " m"),

    html.Div("Semi-altura c  [m]", style=LABEL),
    slider("c", C.DOMINIO.c_hat[0] * C.L_REF, C.DOMINIO.c_hat[1] * C.L_REF, 0.10, 0.005, " m"),

    html.Div("Carga máx. P_max  [kN/m]", style=LABEL),
    slider("P", C.DOMINIO.p_hat[0] * C.P_REF / 1e3, C.DOMINIO.p_hat[1] * C.P_REF / 1e3, 100, 5, " kN"),

    html.Div("Instante de impacto  t̂_imp  (adim.)", style=LABEL),
    slider("timp", C.DOMINIO.t_imp[0], C.DOMINIO.t_imp[1], 1.0, 0.05),

    html.Button("▶  Simular Dinámica", id="simular", n_clicks=0, style={
        "width": "100%", "marginTop": "22px", "padding": "14px",
        "backgroundColor": ACCENT, "color": "#001018", "border": "none",
        "borderRadius": "10px", "fontWeight": 700, "fontSize": "16px", "cursor": "pointer"}),

    html.Div(id="estado", style={"color": MUTED, "fontSize": "12px", "marginTop": "12px"}),
])

viz = html.Div(children=[
    html.Div(style=CARD, children=[
        html.H4("Malla deformada · Tensión de Von Mises [MPa]",
                style={"color": TEXT, "marginTop": 0}),
        dcc.Loading(dcc.Graph(id="main-graph", style={"height": "440px"}), color=ACCENT),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
        html.Div(style=CARD, children=[
            html.H4("Oscilación del extremo libre  v(L,0,t)", style={"color": TEXT, "marginTop": 0}),
            dcc.Graph(id="tip-graph", style={"height": "280px"}),
        ]),
        html.Div(style=CARD, children=[
            html.H4("σ_xx a lo largo de la viga (y=0)", style={"color": TEXT, "marginTop": 0}),
            dcc.Graph(id="sxx-graph", style={"height": "240px"}),
            html.Div("Instante de visualización", style={"color": MUTED, "fontSize": "12px"}),
            dcc.Slider(id="time-slider", min=0, max=1, value=0, step=1,
                       tooltip={"placement": "bottom", "always_visible": False}),
        ]),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
        html.Div(style=CARD, children=[
            html.H4("📐 Métricas de simulación", style={"color": TEXT, "marginTop": 0}),
            html.Div(id="sim-metrics", children=html.Div(
                "Pulsa «Simular Dinámica» para calcular.", style={"color": MUTED, "fontSize": "13px"})),
        ]),
        html.Div(style=CARD, children=[
            html.H4("🔬 Validación física", style={"color": TEXT, "marginTop": 0}),
            html.Div(id="valid-metrics", children=html.Div(
                "Pulsa «Simular Dinámica» para validar.", style={"color": MUTED, "fontSize": "13px"})),
        ]),
    ]),
])

app.layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh", "padding": "24px",
                             "fontFamily": "Inter, system-ui, sans-serif"}, children=[
    html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px", "marginBottom": "20px"}, children=[
        html.H1("🔧 Elastodinámica de Viga 2D", style={"color": TEXT, "margin": 0}),
        html.Span("PINNsFormer · Tensión Plana · Navier-Cauchy",
                  style={"color": ACCENT, "fontSize": "14px", "fontWeight": 600}),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "340px 1fr", "gap": "20px"},
             children=[html.Div([controls, _training_card(CKPT)]), viz]),
    dcc.Store(id="store"),
])


# ----------------------------------------------------------------------
# Callback principal: simulación e inferencia
# ----------------------------------------------------------------------
@app.callback(
    Output("main-graph", "figure"), Output("tip-graph", "figure"),
    Output("store", "data"), Output("time-slider", "max"),
    Output("time-slider", "value"), Output("estado", "children"),
    Output("sim-metrics", "children"), Output("valid-metrics", "children"),
    Input("simular", "n_clicks"),
    State("L", "value"), State("c", "value"), State("P", "value"),
    State("timp", "value"), State("material", "value"),
    prevent_initial_call=False,
)
def simular(_, L, c, P_kN, t_imp, material_key):
    if MODEL is None:
        msg = "⚠️ Modelo no disponible. Entrena en RunPod y sube el checkpoint (ver README)."
        warn = html.Div("Sin modelo.", style={"color": MUTED})
        return _empty_fig("Sin modelo"), _empty_fig(""), None, 1, 0, msg, warn, warn

    import time
    t0 = time.time()
    mat = C.MATERIALES[material_key]
    P_max = P_kN * 1e3
    data = evaluate_fields(MODEL, mat, L=L, c=c, P_max=P_max, t_imp_hat=t_imp, device=DEVICE)

    # Factor de exageración de la deformada (se muestra para honestidad visual).
    max_disp = max(np.abs(data["u"]).max(), np.abs(data["v"]).max(), 1e-12)
    scale = 0.18 * L / max_disp
    t_imp_ms = t_imp * mat.T_ref * 1e3

    main_fig = _build_main_figure(data, L, c, scale)
    tip_fig = _build_tip_figure(data, t_imp_ms)

    # --- Métricas y validación (docs/03 §12 y §13.2) ---
    sim = simulation_metrics(data, mat)
    sim["sanity"] = sanity_checks(data)     # Nivel 2 (siempre activo)
    sim_card = _sim_metrics_children(sim)
    valid_card = _validation_children(sim, mat, L, c, P_max, t_imp)

    # Datos ligeros para el slider de σ_xx (se actualiza sin reinferir)
    store = {
        "x_line": (data["xi"] * L).tolist(),
        "sxx_y0": (data["sxx"][:, data["j0"], :] / 1e6).tolist(),  # MPa
        "t_phys_ms": (data["t_phys"] * 1e3).tolist(),
    }
    nt = len(data["t_hat"])
    dt_ms = (time.time() - t0) * 1e3
    estado = (f"✅ Inferencia en {dt_ms:.0f} ms · {mat.nombre} · "
              f"L={L:.2f} m, c={c:.3f} m, P={P_kN:.0f} kN/m · "
              f"deformada ×{scale:,.0f}")
    return main_fig, tip_fig, store, nt - 1, 0, estado, sim_card, valid_card


def _sim_metrics_children(sim):
    """Construye el contenido de la tarjeta de métricas de simulación."""
    safety = sim["safety"]
    verdict = html.Div(safety["veredicto"], style={
        "color": TONE[safety["tone"]], "fontWeight": 700, "fontSize": "14px",
        "margin": "4px 0 10px"})
    return [verdict, _rows_table(sim["rows"])]


def _validation_children(sim, mat, L, c, P_max, t_imp):
    """Construye el contenido de la tarjeta de validación física (§12),
    en orden didáctico: Nivel 2 → Nivel 3 → Nivel 4."""
    def _subtitle(txt):
        return html.Div(txt, style={"color": MUTED, "fontSize": "12px", "marginTop": "10px"})

    blocks = []

    # NIVEL 2 — sanity-checks (siempre activo). Se calculan en el callback
    # principal (necesitan los campos `data`) y se inyectan en sim["sanity"].
    if "sanity" in sim:
        blocks.append(_subtitle("Nivel 2 · Comprobaciones físicas"))
        blocks.append(_checks_table(sim["sanity"]))

    # NIVEL 3 — referencia analítica de Euler-Bernoulli (conmutable).
    if C.VALIDACION.referencia_analitica:
        blocks.append(_subtitle("Nivel 3 · Contraste Euler-Bernoulli"))
        blocks.append(_checks_table(compare_with_reference(sim, mat, L, c, P_max)))
    else:
        blocks.append(html.Div("Nivel 3 desactivado (config.VALIDACION.referencia_analitica=False).",
                               style={"color": MUTED, "fontSize": "11px", "marginTop": "10px"}))

    # NIVEL 4 — residuo de la EDP en malla nueva (conmutable, autograd 2.º orden).
    if C.VALIDACION.residuo_en_malla:
        blocks.append(_subtitle("Nivel 4 · Residuo EDP en malla nueva (generaliza)"))
        try:
            res = validation_residual(MODEL, mat, C.DOMINIO, C.ENTRENAMIENTO,
                                      L=L, c=c, P_max=P_max, t_imp=t_imp,
                                      n=C.VALIDACION.n_residuo_check, device=DEVICE)
            ok = res < C.VALIDACION.umbral_loss_warn
            blocks.append(_checks_table([("Residuo medio |r|²", f"{res:.2e}", ok)]))
        except Exception as exc:
            blocks.append(html.Div(f"Residuo no disponible ({exc}).",
                                   style={"color": MUTED, "fontSize": "12px"}))
    return blocks


@app.callback(
    Output("sxx-graph", "figure"),
    Input("time-slider", "value"), Input("store", "data"),
)
def actualizar_sxx(idx, store):
    if not store:
        return _empty_fig("")
    idx = int(idx or 0)
    x = store["x_line"]
    sxx = store["sxx_y0"][idx]
    t_ms = store["t_phys_ms"][idx]
    fig = go.Figure(go.Scatter(x=x, y=sxx, mode="lines", line=dict(color=ACCENT, width=3),
                               fill="tozeroy", fillcolor="rgba(0,212,255,0.12)"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      margin=dict(l=50, r=20, t=30, b=40),
                      title=dict(text=f"t = {t_ms:.3f} ms", font=dict(size=12, color=MUTED)),
                      xaxis_title="x [m]", yaxis_title="σ_xx [MPa]")
    return fig


# ----------------------------------------------------------------------
# Constructores de figuras
# ----------------------------------------------------------------------
def _empty_fig(text):
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      annotations=[dict(text=text, showarrow=False,
                                        font=dict(color=MUTED, size=14))])
    return fig


def _build_main_figure(data, L, c, scale):
    """Malla deformada animada coloreada por Von Mises (MPa).

    `scale` es el factor de exageración de la deformada (se calcula en el
    callback y se muestra al usuario para que sepa que está magnificada).
    """
    x, y = data["x"], data["y"]
    u, v = data["u"], data["v"]
    von = data["von_mises"] / 1e6  # MPa
    nt = x.shape[0]
    vmax = max(von.max(), 1e-9)

    def frame_trace(i):
        xd = (x[i] + scale * u[i]).ravel()
        yd = (y[i] + scale * v[i]).ravel()
        return go.Scatter(
            x=xd, y=yd, mode="markers",
            marker=dict(size=7, color=von[i].ravel(), colorscale="Turbo",
                        cmin=0, cmax=vmax, colorbar=dict(title="MPa"), symbol="square"),
            hovertemplate="x=%{x:.3f} m<br>y=%{y:.4f} m<extra></extra>")

    frames = [go.Frame(data=[frame_trace(i)], name=str(i)) for i in range(nt)]
    fig = go.Figure(data=[frame_trace(0)], frames=frames)

    play = dict(label="▶ Play", method="animate",
                args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)])
    pause = dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
    steps = [dict(method="animate", label=f"{data['t_phys'][i]*1e3:.2f}",
                  args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
             for i in range(nt)]

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        margin=dict(l=40, r=20, t=20, b=20),
        xaxis=dict(title="x [m]", range=[-0.05 * L, 1.15 * L], zeroline=False),
        yaxis=dict(title="y [m]", scaleanchor="x", scaleratio=1.0, zeroline=False),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.12,
                          direction="left", buttons=[play, pause])],
        sliders=[dict(active=0, x=0.12, len=0.85, y=0, currentvalue=dict(prefix="t = ", suffix=" ms"),
                      steps=steps)],
    )
    return fig


def _build_tip_figure(data, t_imp_ms=None):
    t_ms = data["t_phys"] * 1e3
    tip = data["tip_v"] * 1e6  # micras
    fig = go.Figure(go.Scatter(x=t_ms, y=tip, mode="lines",
                               line=dict(color="#ff6b6b", width=3)))
    # Marca vertical del instante de impacto (referencia temporal del golpe).
    if t_imp_ms is not None:
        fig.add_vline(x=t_imp_ms, line=dict(color=ACCENT, width=1.5, dash="dash"),
                      annotation_text="impacto", annotation_position="top",
                      annotation_font_color=ACCENT)
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      margin=dict(l=55, r=20, t=20, b=40),
                      xaxis_title="t [ms]", yaxis_title="v(L,0,t) [µm]")
    return fig


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    app.run(debug=False, host="0.0.0.0", port=port)
