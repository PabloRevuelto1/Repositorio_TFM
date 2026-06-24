"""
utils/metrics.py
================
Métricas numéricas y validación física del modelo entrenado.

Implementa los dos bloques planificados en docs/03:

  * §13.2 — MÉTRICAS DE SIMULACIÓN: escalares cuantitativos derivados de los
    campos ya evaluados (flecha, periodo de oscilación, tensiones máximas,
    deformaciones, factor de seguridad...).

  * §12   — VALIDACIÓN FÍSICA en cuatro niveles:
        N1 (pérdidas finales)      → se leen del checkpoint (train.py).
        N2 (sanity-checks)         → `sanity_checks()`        [siempre activo].
        N3 (referencia analítica)  → `reference_euler_bernoulli()` y
                                      `compare_with_reference()` [conmutable].
        N4 (residuo en malla)      → `validation_residual()`   [conmutable].

Todas las funciones operan sobre el dict que devuelve
`utils.inference.evaluate_fields` y/o el modelo cargado; ninguna reentrena.
"""

import math

import numpy as np
import torch

import config as C


# ======================================================================
# §13.2 — MÉTRICAS DE SIMULACIÓN
# ======================================================================
def _dominant_frequency(signal, dt):
    """Frecuencia dominante [Hz] de una señal 1D por FFT (tras quitar la media).

    Devuelve (frecuencia_Hz, periodo_s) o (nan, nan) si no es resoluble.
    """
    sig = np.asarray(signal, dtype=float)
    sig = sig - sig.mean()
    n = sig.size
    if n < 4 or not np.any(np.abs(sig) > 0):
        return math.nan, math.nan
    # FFT real; se ignora la componente DC (índice 0).
    amp = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, d=dt)
    k = 1 + int(np.argmax(amp[1:]))
    f = float(freqs[k])
    if f <= 0:
        return math.nan, math.nan
    return f, 1.0 / f


def _argmax_location(field3d, x_phys, y_phys):
    """Posición física (x, y) [m] del máximo en valor absoluto de un campo (nt,ny,nx)."""
    idx = np.unravel_index(np.argmax(np.abs(field3d)), field3d.shape)
    return float(x_phys[idx]), float(y_phys[idx])


def simulation_metrics(data, material):
    """
    Escalares cuantitativos de una simulación (dict de `evaluate_fields`).

    Devuelve un dict:
        {
          "rows":   [(etiqueta, valor_formateado), ...],   # para tabla
          "safety": {"factor": float, "tone": str, "veredicto": str},
          "raw":    {...}                                   # valores numéricos
        }
    """
    u, v = data["u"], data["v"]
    sxx, von = data["sxx"], data["von_mises"]
    exx, eyy, exy = data["exx"], data["eyy"], data["exy"]
    x_phys, y_phys = data["x"], data["y"]
    t_phys, tip_v = data["t_phys"], data["tip_v"]

    # --- Desplazamientos / flecha ---
    disp_mag = np.sqrt(u ** 2 + v ** 2)
    disp_max = float(disp_mag.max())
    tip_abs_max = float(np.abs(tip_v).max())
    tip_static = float(np.mean(tip_v))    # ≈ flecha estática (oscila en torno a ella)

    # --- Oscilación del extremo libre ---
    dt = float(t_phys[1] - t_phys[0]) if t_phys.size > 1 else math.nan
    f_osc, T_osc = _dominant_frequency(tip_v, dt)

    # --- Tensiones ---
    sxx_max = float(np.abs(sxx).max())
    sxx_xy = _argmax_location(sxx, x_phys, y_phys)
    von_max = float(von.max())
    von_xy = _argmax_location(von, x_phys, y_phys)

    # --- Deformaciones (adimensionales, ~µε) ---
    eps_max = float(np.max(np.abs([exx, eyy, exy])))

    # --- Factor de seguridad (cierra el círculo con docs/03 §7) ---
    fs = material.sigma_y / von_max if von_max > 0 else math.inf
    fs_obj = C.VALIDACION.coef_seguridad
    if von_max >= material.sigma_y:
        tone, veredicto = "bad", "⚠️ PLASTIFICA (σ_VM ≥ σ_y)"
    elif fs < fs_obj:
        tone, veredicto = "warn", f"elástico, pero F.S. < {fs_obj:g}"
    else:
        tone, veredicto = "ok", f"✅ elástico y seguro (F.S. ≥ {fs_obj:g})"

    rows = [
        ("Flecha máx. extremo |v(L,0,t)|", f"{tip_abs_max * 1e6:.2f} µm"),
        ("Flecha estática (media temporal)", f"{tip_static * 1e6:.2f} µm"),
        ("Frecuencia de oscilación", f"{f_osc / 1e3:.2f} kHz" if not math.isnan(f_osc) else "n/d"),
        ("Periodo de oscilación", f"{T_osc * 1e3:.4f} ms" if not math.isnan(T_osc) else "n/d"),
        ("Desplazamiento máx. global", f"{disp_max * 1e6:.2f} µm"),
        ("σ_xx máx.", f"{sxx_max / 1e6:.1f} MPa @ ({sxx_xy[0]:.2f}, {sxx_xy[1]:+.3f}) m"),
        ("σ_VM máx.", f"{von_max / 1e6:.1f} MPa @ ({von_xy[0]:.2f}, {von_xy[1]:+.3f}) m"),
        ("Deformación máx. |ε|", f"{eps_max * 1e6:.1f} µε"),
        ("Límite elástico σ_y", f"{material.sigma_y / 1e6:.0f} MPa"),
        ("Factor de seguridad σ_y/σ_VM", f"{fs:.2f}"),
    ]
    return {
        "rows": rows,
        "safety": {"factor": fs, "tone": tone, "veredicto": veredicto},
        "raw": {"tip_static": tip_static, "f_osc": f_osc, "von_max": von_max},
    }


# ======================================================================
# §12 · NIVEL 2 — SANITY CHECKS (siempre activo, prácticamente gratis)
# ======================================================================
def sanity_checks(data):
    """
    Comprobaciones físicas que DEBEN cumplirse. Devuelve lista de
    (etiqueta, valor_formateado, ok_bool).
    """
    u, v = data["u"], data["v"]
    syy, txy = data["syy"], data["txy"]

    # IC: en t=0 la viga debe estar plana (índice temporal 0).
    ic_disp = float(np.sqrt(u[0] ** 2 + v[0] ** 2).max())
    # Empotramiento: el extremo ξ=0 (primer índice de x) no se mueve nunca.
    clamp_disp = float(np.sqrt(u[:, :, 0] ** 2 + v[:, :, 0] ** 2).max())
    # Superficies libres η=±1 (primer/último índice de y): σ_yy, τ_xy ≈ 0.
    free_sig = float(np.abs(np.concatenate([syy[:, 0], syy[:, -1],
                                            txy[:, 0], txy[:, -1]])).max())

    # Umbrales tolerantes (la imposición es blanda; ver docs/03 §11.1).
    tol_disp = 1e-6      # 1 µm
    tol_sig = 5e6        # 5 MPa
    return [
        ("Reposo en t=0  (máx. |d|)", f"{ic_disp * 1e6:.2f} µm", ic_disp < tol_disp),
        ("Empotramiento fijo  (máx. |d| en x=0)", f"{clamp_disp * 1e6:.2f} µm", clamp_disp < tol_disp),
        ("Superficies libres  (máx. |σ_yy|,|τ_xy|)", f"{free_sig / 1e6:.1f} MPa", free_sig < tol_sig),
    ]


# ======================================================================
# §12 · NIVEL 3 — REFERENCIA ANALÍTICA (Euler-Bernoulli) [conmutable]
# ======================================================================
def reference_euler_bernoulli(material, L, c, P_max):
    """
    Solución analítica de una ménsula (cantilever) con carga cortante P en la
    punta, por unidad de espesor. Coste despreciable (fórmula cerrada).

        I  = 2 c³ / 3            (momento de inercia, espesor unidad)
        δ  = P L³ / (3 E I)      (flecha estática del extremo)
        f₁ = (β₁)² /(2π L²) · √(E I /(ρ A))   con β₁ = 1.875104  (1ª frec. natural)

    Devuelve dict {delta [m], f1 [Hz]}.
    """
    E, rho = material.E, material.rho
    I = 2.0 * c ** 3 / 3.0
    A = 2.0 * c
    delta = P_max * L ** 3 / (3.0 * E * I)
    beta1 = 1.875104
    f1 = (beta1 ** 2) / (2.0 * math.pi * L ** 2) * math.sqrt(E * I / (rho * A))
    return {"delta": delta, "f1": f1}


def compare_with_reference(sim_metrics, material, L, c, P_max):
    """
    Compara la simulación PINN con la referencia de Euler-Bernoulli (Nivel 3).
    Devuelve lista de (etiqueta, valor_formateado, ok_bool).
    """
    ref = reference_euler_bernoulli(material, L, c, P_max)
    raw = sim_metrics["raw"]

    def err(pinn, teo):
        return abs(pinn - teo) / abs(teo) * 100.0 if teo else math.nan

    e_delta = err(abs(raw["tip_static"]), ref["delta"])
    rows = [
        ("Flecha estática — Euler-Bernoulli", f"{ref['delta'] * 1e6:.2f} µm", None),
        ("Flecha estática — PINN", f"{abs(raw['tip_static']) * 1e6:.2f} µm", None),
        ("Error relativo flecha", f"{e_delta:.1f} %", e_delta < 20.0),
        ("1ª frecuencia natural — Euler-Bernoulli", f"{ref['f1'] / 1e3:.2f} kHz", None),
    ]
    if not math.isnan(raw["f_osc"]):
        e_f = err(raw["f_osc"], ref["f1"])
        rows.append(("Frecuencia — PINN", f"{raw['f_osc'] / 1e3:.2f} kHz", None))
        rows.append(("Error relativo frecuencia", f"{e_f:.1f} %", e_f < 30.0))
    return rows


# ======================================================================
# §12 · NIVEL 4 — RESIDUO DE LA EDP EN MALLA DE INFERENCIA [conmutable]
# ======================================================================
def validation_residual(model, material, dom, ent, n=400, device="cpu", chunk=256):
    """
    Evalúa el residuo medio de Navier-Cauchy sobre `n` puntos interiores del
    escenario fijo (puntos NO vistos en entrenamiento).

    Reutiliza `ElastodynamicsLoss.loss_residual` (autograd de 2.º orden). Para
    NO agotar la RAM al ejecutar la app en CPU/WSL (la diferenciación de 2.º
    orden es justo la operación pesada que el README desaconseja en WSL), el
    cálculo se hace POR BLOQUES (`chunk`) promediando el residuo ponderado por
    el tamaño de cada bloque. Devuelve el residuo cuadrático medio (float).
    """
    from physics.pde_loss import ElastodynamicsLoss
    from utils.sampling import _lhs, make_time_sequence

    # LHS sobre (ξ, η, t̂) — escenario fijo, sin entradas paramétricas.
    bounds = [dom.xi, dom.eta, dom.t]
    pts = _lhs(n, bounds, seed=ent.seed + 99)
    seq = make_time_sequence(pts, k=C.MODELO.k, step=C.MODELO.seq_step).astype(np.float32)

    loss_fn = ElastodynamicsLoss(model, material, dom, ent)

    total = 0.0
    for s in range(0, n, chunk):
        sub = seq[s:s + chunk]
        batch = torch.tensor(sub, dtype=torch.float32, device=device)
        with torch.enable_grad():
            res = loss_fn.loss_residual(batch)
        total += float(res.detach().cpu()) * (sub.shape[0] / n)
        del batch, res
    return total


# ======================================================================
# GROUND TRUTH — COMPARACIÓN PINN vs ELEMENTOS FINITOS (scikit-fem)
# ======================================================================
def _rel_l2(a, b):
    """Error L2 relativo (%) de `a` respecto de la referencia `b`."""
    num = float(np.sqrt(np.sum((a - b) ** 2)))
    den = float(np.sqrt(np.sum(b ** 2))) + 1e-30
    return 100.0 * num / den


def compare_fields_pinn_fem(pinn, fem):
    """
    Concordancia campo a campo de la PINN contra el FEM (verdad de referencia),
    en la malla común (nt, ny, nx). Devuelve filas (etiqueta, valor, ok_bool).

    El FEM es el *ground truth*: estas cifras son el "boletín de notas" de la PINN.
    """
    dmag_p = np.sqrt(pinn["u"] ** 2 + pinn["v"] ** 2)
    dmag_f = np.sqrt(fem["u"] ** 2 + fem["v"] ** 2)
    e_disp = _rel_l2(dmag_p, dmag_f)
    e_von = _rel_l2(pinn["von_mises"], fem["von_mises"])
    e_sxx = _rel_l2(pinn["sxx"], fem["sxx"])
    tol = 30.0   # % de error L2 admisible para "buen acuerdo"
    return [
        ("Error L2 desplazamiento", f"{e_disp:.1f} %", e_disp < tol),
        ("Error L2 σ_VM", f"{e_von:.1f} %", e_von < tol),
        ("Error L2 σ_xx", f"{e_sxx:.1f} %", e_sxx < tol),
    ]


def _better(pinn_val, fem_val, ref, lower_better=True):
    """Devuelve 'PINN'/'FEM'/'empate' según quién se acerca más a `ref`."""
    dp, df = abs(pinn_val - ref), abs(fem_val - ref)
    if abs(dp - df) / (abs(ref) + 1e-30) < 0.02:
        return "empate"
    return "PINN" if (dp < df) == lower_better else "FEM"


def winner_table(pinn, fem, material, t_pinn_s, t_fem_s):
    """
    Tabla comparativa de 4 columnas (categoría, PINN, FEM, ganador) que resume
    ventajas/desventajas de cada método. El FEM es la verdad física de referencia;
    la PINN aporta rapidez de consulta y continuidad (malla-free, derivadas exactas).

    Devuelve lista de tuplas (categoria, valor_pinn, valor_fem, ganador).
    """
    p = simulation_metrics(pinn, material)["raw"]
    f = simulation_metrics(fem, material)["raw"]

    # Picos físicos (FEM = referencia → su ubicación de σ_VM en el empotramiento
    # es la "correcta"; se premia a quien la reproduce).
    von_p, von_f = pinn["von_mises"].max(), fem["von_mises"].max()
    tip_p = float(np.abs(pinn["tip_v"]).max())
    tip_f = float(np.abs(fem["tip_v"]).max())

    # Localización de σ_VM máx (¿empotramiento x≈0?). ix==0 es el empotramiento.
    def _ix_vonmax(d):
        return int(np.unravel_index(np.argmax(d["von_mises"]), d["von_mises"].shape)[2])
    ix_p, ix_f = _ix_vonmax(pinn), _ix_vonmax(fem)
    nx = pinn["von_mises"].shape[2]
    loc_p = "empotramiento" if ix_p <= 1 else ("extremo" if ix_p >= nx - 2 else "centro")
    loc_f = "empotramiento" if ix_f <= 1 else ("extremo" if ix_f >= nx - 2 else "centro")

    rows = [
        # (categoría, PINN, FEM, ganador)
        ("Tiempo de cómputo",
         f"{t_pinn_s * 1e3:.0f} ms", f"{t_fem_s * 1e3:.0f} ms",
         "PINN" if t_pinn_s < t_fem_s else "FEM"),
        ("Flecha máx. extremo |v|",
         f"{tip_p * 1e6:.2f} µm", f"{tip_f * 1e6:.2f} µm",
         _better(tip_p, tip_f, tip_f)),   # ref=FEM → mide desvío de la PINN
        ("σ_VM máx.",
         f"{von_p / 1e6:.2f} MPa", f"{von_f / 1e6:.2f} MPa",
         _better(von_p, von_f, von_f)),
        ("Ubicación σ_VM máx.",
         loc_p, loc_f,
         "empate" if loc_p == loc_f else ("FEM" if loc_f == "empotramiento" else "PINN")),
        ("Fidelidad física", "aproximada", "referencia", "FEM"),
        ("Consulta malla-free / derivadas", "sí (autograd)", "no (discreta)", "PINN"),
    ]
    return rows
