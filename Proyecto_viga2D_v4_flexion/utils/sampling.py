"""
utils/sampling.py
=================
Muestreo del dominio paramétrico (Latin Hypercube Sampling) y construcción
de la pseudo-secuencia temporal de PINNsFormer.

Cada conjunto de puntos se devuelve como un diccionario de columnas
adimensionales:  xi, eta, t, L_hat, c_hat, p_hat, t_imp.

La pseudo-secuencia (make_time_sequence) replica cada punto `k` veces
desplazando ÚNICAMENTE la coordenada temporal t̂ un paso `seq_step`,
exactamente como en el repositorio oficial (allí el tiempo es la última
columna; aquí es la columna de índice 2).
"""

import numpy as np
from scipy.stats import qmc

# Orden canónico de las columnas de un lote de puntos (escenario fijo: solo
# coordenadas espacio-temporales; la geometría/carga es constante, ver config.VIGA).
COLS = ["xi", "eta", "t"]
I_T = COLS.index("t")  # índice de la columna temporal


def _lhs(n, bounds, seed=None):
    """Genera `n` muestras Latin Hypercube en los `bounds` dados (lista de (lo, hi)).
    Las dimensiones degeneradas (lo == hi) se fijan a su valor constante."""
    dim = len(bounds)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    sample = sampler.random(n)                       # (n, dim) en [0,1]
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    fixed = np.isclose(lo, hi)
    span = np.where(fixed, 1.0, hi - lo)             # evita división/validación nula
    out = lo + sample * span
    out[:, fixed] = lo[fixed]                         # fija las dimensiones constantes
    return out


def make_time_sequence(arr, k=5, step=1e-3):
    """
    Construye la pseudo-secuencia temporal de PINNsFormer.

    arr : (N, n_feat)  con la columna temporal en el índice I_T.
    ->    (N, k, n_feat) replicando cada punto k veces y sumando
          step*i a la coordenada temporal del instante i.
    """
    seq = np.repeat(arr[:, None, :], k, axis=1)      # (N, k, n_feat)
    for i in range(k):
        seq[:, i, I_T] += step * i
    return seq


def _t_bounds(dom, t_max):
    """Rango temporal de muestreo, opcionalmente recortado a [0, t_max] (currículo)."""
    return (dom.t[0], dom.t[1] if t_max is None else t_max)


def sample_interior(dom, n, seed=None, t_max=None):
    """Puntos de colocación del residuo en el interior del dominio."""
    bounds = [dom.xi, dom.eta, _t_bounds(dom, t_max)]
    return _lhs(n, bounds, seed)


def sample_boundary(dom, n, fixed, seed=None, t_max=None):
    """
    Puntos en una frontera espacial. `fixed` fija coordenadas concretas:
    p.ej. {'xi': 0.0} (empotramiento) o {'eta': 1.0} (superficie superior).
    El tiempo se muestrea por LHS (recortado a [0, t_max] si se da, para el currículo).
    """
    bounds = [dom.xi, dom.eta, _t_bounds(dom, t_max)]
    s = _lhs(n, bounds, seed)
    name_to_col = {c: j for j, c in enumerate(COLS)}
    for name, val in fixed.items():
        s[:, name_to_col[name]] = val
    return s


def sample_initial(dom, n, seed=None):
    """Puntos en la condición inicial (t̂ = 0); espacio por LHS."""
    bounds = [dom.xi, dom.eta, (0.0, 0.0)]
    return _lhs(n, bounds, seed)


def sample_sections(dom, n_sec, n_eta, seed=None, t_max=None):
    """
    Secciones para el EQUILIBRIO INTEGRAL (cortante/momento por sección).

    Muestrea `n_sec` pares (ξ, t̂) por LHS y, para cada uno, coloca una rejilla
    FIJA de `n_eta` valores de η ∈ [-1, 1] (la misma para todas las secciones), de
    modo que ∫·dη se calcula por cuadratura trapezoidal sobre esa rejilla. Las filas
    van AGRUPADAS por sección: la sección i ocupa las filas [i·n_eta : (i+1)·n_eta].
    Devuelve (n_sec·n_eta, 3) en columnas (ξ, η, t̂).
    """
    pares = _lhs(n_sec, [dom.xi, _t_bounds(dom, t_max)], seed)   # (n_sec, 2): ξ, t̂
    eta_grid = np.linspace(dom.eta[0], dom.eta[1], n_eta)
    xi = np.repeat(pares[:, 0], n_eta)
    t = np.repeat(pares[:, 1], n_eta)
    eta = np.tile(eta_grid, n_sec)
    return np.stack([xi, eta, t], axis=-1)               # (n_sec·n_eta, 3)


def build_training_sets(dom, ent, k, step, t_max=None):
    """
    Genera todos los conjuntos de entrenamiento (interior, fronteras, inicial,
    secciones) ya convertidos a pseudo-secuencias (N, k, 3).

    `t_max` (currículo temporal) recorta el muestreo a t̂∈[0, t_max] en todos los
    conjuntos salvo la CI (que vive en t̂=0). Devuelve un dict de arrays float32.
    """
    rng = ent.seed
    sets = {
        "res":      sample_interior(dom, ent.n_res, seed=rng, t_max=t_max),
        "clamp":    sample_boundary(dom, ent.n_bc, {"xi": dom.xi[0]}, seed=rng + 1, t_max=t_max),
        "load":     sample_boundary(dom, ent.n_bc, {"xi": dom.xi[1]}, seed=rng + 2, t_max=t_max),
        "top":      sample_boundary(dom, ent.n_bc, {"eta": dom.eta[1]}, seed=rng + 3, t_max=t_max),
        "bottom":   sample_boundary(dom, ent.n_bc, {"eta": dom.eta[0]}, seed=rng + 4, t_max=t_max),
        "ic":       sample_initial(dom, ent.n_ic, seed=rng + 5),
        "eq":       sample_sections(dom, ent.n_sec, ent.n_eta_eq, seed=rng + 6, t_max=t_max),
    }
    return {name: make_time_sequence(a, k=k, step=step).astype(np.float32)
            for name, a in sets.items()}
