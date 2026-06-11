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

# Orden canónico de las columnas de un lote de puntos.
COLS = ["xi", "eta", "t", "L_hat", "c_hat", "p_hat", "t_imp"]
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


def _param_bounds(dom):
    """Rangos de los 4 parámetros (L̂, ĉ, p̂, t̂_imp)."""
    return [dom.L_hat, dom.c_hat, dom.p_hat, dom.t_imp]


def sample_interior(dom, n, seed=None):
    """Puntos de colocación del residuo en el interior del dominio."""
    bounds = [dom.xi, dom.eta, dom.t] + _param_bounds(dom)
    return _lhs(n, bounds, seed)


def sample_boundary(dom, n, fixed, seed=None):
    """
    Puntos en una frontera espacial. `fixed` fija coordenadas concretas:
    p.ej. {'xi': 0.0} (empotramiento) o {'eta': 1.0} (superficie superior).
    El tiempo y los parámetros se muestrean por LHS.
    """
    bounds = [dom.xi, dom.eta, dom.t] + _param_bounds(dom)
    s = _lhs(n, bounds, seed)
    name_to_col = {c: j for j, c in enumerate(COLS)}
    for name, val in fixed.items():
        s[:, name_to_col[name]] = val
    return s


def sample_initial(dom, n, seed=None):
    """Puntos en la condición inicial (t̂ = 0); espacio y parámetros por LHS."""
    bounds = [dom.xi, dom.eta, (0.0, 0.0)] + _param_bounds(dom)
    return _lhs(n, bounds, seed)


def build_training_sets(dom, ent, k, step):
    """
    Genera todos los conjuntos de entrenamiento (interior, fronteras, inicial)
    ya convertidos a pseudo-secuencias (N, k, 7).

    Devuelve un dict de arrays numpy float32.
    """
    rng = ent.seed
    sets = {
        "res":      sample_interior(dom, ent.n_res, seed=rng),
        "clamp":    sample_boundary(dom, ent.n_bc, {"xi": dom.xi[0]}, seed=rng + 1),
        "load":     sample_boundary(dom, ent.n_bc, {"xi": dom.xi[1]}, seed=rng + 2),
        "top":      sample_boundary(dom, ent.n_bc, {"eta": dom.eta[1]}, seed=rng + 3),
        "bottom":   sample_boundary(dom, ent.n_bc, {"eta": dom.eta[0]}, seed=rng + 4),
        "ic":       sample_initial(dom, ent.n_ic, seed=rng + 5),
    }
    return {name: make_time_sequence(a, k=k, step=step).astype(np.float32)
            for name, a in sets.items()}
