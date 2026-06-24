"""
utils/inference.py
==================
Carga del modelo entrenado y evaluación rápida (inferencia paramétrica) de
los campos de desplazamiento y tensión sobre una malla espacio-temporal.

La red trabaja en variables adimensionales; aquí se reconvierten a unidades
físicas según el material seleccionado:

    u_físico = U · U_REF(material)
    σ_físico = σ̂ · SIGMA_SCALE                 (independiente del material)
    t_físico = t̂ · T_REF(material)

No se reentrena nada: se cargan los pesos de checkpoints/best_model.pth y se
realiza una única pasada vectorizada por la red.
"""

import numpy as np
import torch

import config as C
from models.pinnsformer import PINNsformer
from utils.sampling import make_time_sequence


def load_model(checkpoint_path=None, device="cpu"):
    """Carga el modelo PINNsFormer entrenado desde el checkpoint."""
    path = checkpoint_path or C.ENTRENAMIENTO.checkpoint
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_cfg"]

    # Compatibilidad con la versión SIMPLIFICADA (d_in=3). Un checkpoint antiguo de
    # la red paramétrica (d_in=7) cargaría sus pesos pero fallaría en inferencia
    # (forward espera 3 entradas). Se rechaza con un mensaje claro: hay que
    # reentrenar con el código actual.
    if cfg.get("d_in") != C.MODELO.d_in:
        raise ValueError(
            f"checkpoint incompatible: d_in={cfg.get('d_in')} pero el proyecto usa "
            f"d_in={C.MODELO.d_in} (versión simplificada). Reentrena con el código actual.")

    lo, hi = ckpt["input_bounds"]
    model = PINNsformer(
        d_in=cfg["d_in"], d_out=cfg["d_out"], d_model=cfg["d_model"],
        d_hidden=cfg["d_hidden"], N=cfg["N"], heads=cfg["heads"],
        in_lo=lo, in_hi=hi,
        # Restricciones duras: flag, ic_tau y modo de CI viven en el checkpoint (compat.
        # con checkpoints antiguos: hard_bc desactivado, τ por defecto, hard_ic=True que
        # era el comportamiento previo a la CI de velocidad blanda).
        hard_bc=cfg.get("hard_constraints", False),
        ic_tau=cfg.get("ic_tau", 1.0),
        hard_ic=cfg.get("hard_ic", True),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _grad(out, inp):
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                               retain_graph=True, create_graph=False)[0]


def evaluate_fields(model, material=None, nx=45, ny=15, nt=40, device="cpu"):
    """
    Evalúa los campos de la PINN en una malla (nt, ny, nx) para el ESCENARIO
    FIJO (config.VIGA): geometría, carga y perfil temporal constantes.

    Devuelve un dict con arrays numpy de forma (nt, ny, nx):
        x, y           : malla física de referencia [m]
        u, v           : desplazamientos físicos [m]
        sxx, syy, txy  : tensiones físicas [Pa]
        von_mises      : tensión de Von Mises [Pa]
        exx, eyy, exy  : deformaciones físicas (adimensionales)
        t_hat, t_phys  : vectores de tiempo (adim. y físico [s])
        tip_v          : desplazamiento vertical del extremo (x=L,y=0) [m] (nt,)
    """
    mod = C.MODELO
    material = material or C.MATERIAL
    L, c = C.VIGA.L, C.VIGA.c

    # Malla normalizada
    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(-1.0, 1.0, ny)
    t_hat = np.linspace(0.0, C.DOMINIO.T_hat, nt)
    TT, EE, XX = np.meshgrid(t_hat, eta, xi, indexing="ij")  # (nt,ny,nx)

    base = np.stack([XX.ravel(), EE.ravel(), TT.ravel()], axis=-1).astype(np.float32)

    # Evaluación por bloques (chunks) para acotar el uso de memoria: la app
    # se ejecuta también en equipos modestos / WSL, donde una única pasada con
    # decenas de miles de puntos podría agotar la RAM. Sólo se usan derivadas
    # de primer orden (create_graph=False), mucho más ligeras que el residuo.
    seq_full = make_time_sequence(base, k=mod.k, step=mod.seq_step).astype(np.float32)
    chunk = 4096
    cols = {k: [] for k in ("U", "V", "Ux", "Ue", "Vx", "Ve")}
    for s in range(0, seq_full.shape[0], chunk):
        seq = torch.tensor(seq_full[s:s + chunk], dtype=torch.float32, device=device)
        xi_t = seq[:, :, 0:1].clone().requires_grad_(True)
        eta_t = seq[:, :, 1:2].clone().requires_grad_(True)
        t_t = seq[:, :, 2:3]
        with torch.enable_grad():
            out = model(xi_t, eta_t, t_t)
            U, V = out[:, :, 0:1], out[:, :, 1:2]
            U_xi, U_eta = _grad(U, xi_t), _grad(U, eta_t)
            V_xi, V_eta = _grad(V, xi_t), _grad(V, eta_t)
        for key, val in zip(cols, (U, V, U_xi, U_eta, V_xi, V_eta)):
            cols[key].append(val[:, 0, 0].detach().cpu().numpy())

    def field(key):
        return np.concatenate(cols[key]).reshape(nt, ny, nx)

    L_hat, c_hat = C.VIGA.L_hat, C.VIGA.c_hat
    U0, V0 = field("U"), field("V")
    u_x = field("Ux") / L_hat
    v_y = field("Ve") / c_hat
    u_y = field("Ue") / c_hat
    v_x = field("Vx") / L_hat

    mu, lam = material.mu_hat, material.lam_hat
    sxx_h = (lam + 2 * mu) * u_x + lam * v_y      # tensiones adimensionales
    syy_h = lam * u_x + (lam + 2 * mu) * v_y
    txy_h = mu * (u_y + v_x)

    # A unidades físicas
    U_ref = material.U_ref
    u = U0 * U_ref
    v = V0 * U_ref
    sxx = sxx_h * C.SIGMA_SCALE
    syy = syy_h * C.SIGMA_SCALE
    txy = txy_h * C.SIGMA_SCALE
    von = np.sqrt(np.clip(sxx ** 2 - sxx * syy + syy ** 2 + 3 * txy ** 2, 0, None))

    # Deformaciones físicas (adimensionales). A partir de las adimensionales
    # u_x, v_y, ... el paso a físicas es el factor σ_SCALE/E (ver docs/03 §8).
    eps_fac = C.SIGMA_SCALE / material.E
    exx = u_x * eps_fac
    eyy = v_y * eps_fac
    exy = 0.5 * (u_y + v_x) * eps_fac

    x_phys = (xi * L)[None, None, :] * np.ones((nt, ny, 1))
    y_phys = (eta * c)[None, :, None] * np.ones((nt, 1, nx))

    # Extremo libre (x=L → último índice de ξ, y=0 → eta central)
    j0 = ny // 2
    tip_v = v[:, j0, -1]

    return {
        "x": x_phys, "y": y_phys, "u": u, "v": v,
        "sxx": sxx, "syy": syy, "txy": txy, "von_mises": von,
        "exx": exx, "eyy": eyy, "exy": exy,
        "t_hat": t_hat, "t_phys": t_hat * material.T_ref,
        "tip_v": tip_v, "xi": xi, "eta": eta, "j0": j0,
    }
