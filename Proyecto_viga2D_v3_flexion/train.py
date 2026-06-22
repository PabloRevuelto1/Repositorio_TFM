"""
train.py
========
Entrenamiento del modelo PINNsFormer para la elastodinámica 2D de una viga
empotrada bajo carga de impacto.

Flujo:
  1. Muestreo del dominio paramétrico (LHS) + pseudo-secuencia temporal.
  2. Entrenamiento en dos fases (estándar en PINNs):
        Fase A → Adam   (exploración rápida del paisaje de pérdida)
        Fase B → L-BFGS (refinado de alta precisión, strong_wolfe)
  3. Guardado del mejor modelo en checkpoints/best_model.pth junto con la
     configuración necesaria para la inferencia.

Uso:
    python train.py                       # entrenamiento completo
    python train.py --adam 500 --lbfgs 0  # prueba rápida
"""

import argparse
import os

# Reduce la fragmentación del asignador CUDA (debe fijarse ANTES de importar torch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import signal
import time

import numpy as np
import torch
from torch.optim import Adam, LBFGS

import config as C
from models.pinnsformer import PINNsformer, get_n_params, init_weights
from physics.pde_loss import ElastodynamicsLoss
from utils.sampling import build_training_sets


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento PINNsFormer viga 2D")
    p.add_argument("--adam", type=int, default=C.ENTRENAMIENTO.adam_epochs)
    p.add_argument("--lbfgs", type=int, default=C.ENTRENAMIENTO.lbfgs_epochs)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--chunk", type=int, default=512,
                   help="tamaño de bloque para acumular gradiente (acota la VRAM)")
    p.add_argument("--log-every", type=int, default=25,
                   help="cada cuántas épocas se imprime el progreso")
    p.add_argument("--max-minutes", type=float, default=0.0,
                   help="presupuesto de tiempo; al superarse se detiene y se guarda (0 = sin límite)")
    return p.parse_args()


def to_tensor(sets, device):
    """Convierte el dict de arrays numpy a tensores en el dispositivo."""
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in sets.items()}


def main():
    args = parse_args()
    ent, mod, dom = C.ENTRENAMIENTO, C.MODELO, C.DOMINIO

    # Reproducibilidad
    np.random.seed(ent.seed)
    torch.manual_seed(ent.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] dispositivo: {device}")

    material = C.MATERIALES[C.MATERIAL_ENTRENAMIENTO]
    print(f"[info] material de entrenamiento: {material.nombre} "
          f"(μ̂={material.mu_hat:.3f}, λ̂={material.lam_hat:.3f})")

    # --- Datos ---
    print("[info] muestreando dominio (LHS) y construyendo pseudo-secuencias...")
    sets_np = build_training_sets(dom, ent, k=mod.k, step=mod.seq_step)
    sets = to_tensor(sets_np, device)
    for name, arr in sets_np.items():
        print(f"        {name:8s}: {arr.shape}")

    # --- Modelo ---
    lo, hi = C.input_bounds()
    model = PINNsformer(
        d_in=mod.d_in, d_out=mod.d_out, d_model=mod.d_model,
        d_hidden=mod.d_hidden, N=mod.N, heads=mod.heads,
        in_lo=lo, in_hi=hi,
        hard_bc=mod.hard_constraints, ic_tau=mod.ic_tau,
    ).to(device)
    model.apply(init_weights)
    print(f"[info] parámetros del modelo: {get_n_params(model):,}")

    # --- Pérdida física ---
    loss_fn = ElastodynamicsLoss(model, material, dom, ent)

    best = {"loss": float("inf"), "state": None}
    history = []

    # ------------------------------------------------------------------
    # Backward con ACUMULACIÓN DE GRADIENTE por bloques.
    # El residuo de Navier-Cauchy exige derivadas de SEGUNDO orden a través
    # del transformer; evaluarlo de golpe para miles de puntos agota la VRAM.
    # Procesando en sub-lotes y ponderando por su fracción (frac) se obtiene
    # EXACTAMENTE el mismo gradiente medio, con memoria acotada.
    # ------------------------------------------------------------------
    chunk = args.chunk

    def _accumulate(name, fn, weight, comps_key, comps):
        data = sets[name]
        N = data.shape[0]
        for s in range(0, N, chunk):
            sub = data[s:s + chunk]
            frac = sub.shape[0] / N
            l = fn(sub)
            (weight * l * frac).backward()
            comps[comps_key] += l.item() * frac

    def compute_grads():
        """Pone a cero, acumula gradiente por bloques y devuelve (loss_total, comps)."""
        for p in model.parameters():
            p.grad = None
        comps = {"res": 0.0, "load": 0.0, "bc": 0.0, "ic": 0.0, "eq": 0.0}
        _accumulate("res", loss_fn.loss_residual, ent.w_res, "res", comps)
        _accumulate("load", loss_fn.loss_load, ent.w_load, "load", comps)   # carga: peso propio
        for b, f in (("clamp", loss_fn.loss_clamp),
                     ("top", loss_fn.loss_free_surface), ("bottom", loss_fn.loss_free_surface)):
            _accumulate(b, f, ent.w_bc, "bc", comps)
        _accumulate("ic", loss_fn.loss_initial, ent.w_ic, "ic", comps)
        # Equilibrio seccional: NO se trocea (la integral ∫·dη exige secciones
        # completas). Es de 1.er orden (ligero); un único backward acumula su grad.
        l_eq = loss_fn.loss_equilibrium(sets["eq"])
        (ent.w_eq * l_eq).backward()
        comps["eq"] = l_eq.item()
        total = (ent.w_res * comps["res"] + ent.w_load * comps["load"]
                 + ent.w_bc * comps["bc"] + ent.w_ic * comps["ic"]
                 + ent.w_eq * comps["eq"])
        return total, comps

    def _save_if_best(val, comps):
        if val < best["loss"]:
            best.update(loss=val, comps=dict(comps),
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    # --- Detención controlada: presupuesto de tiempo + señal SIGTERM ---
    # Garantiza que el modelo entrenado HASTA EL MOMENTO se guarde y se suba,
    # aunque RunPod corte el Pod o se agote el tiempo (no se pierde el progreso).
    t_train0 = time.time()
    stop = {"flag": False}

    def _on_sigterm(_signum, _frame):
        print("\n[señal] SIGTERM recibido → deteniendo y guardando el mejor modelo...", flush=True)
        stop["flag"] = True
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        pass  # p. ej. si no se ejecuta en el hilo principal

    def _should_stop():
        if stop["flag"]:
            return True
        if args.max_minutes and (time.time() - t_train0) / 60.0 >= args.max_minutes:
            print(f"\n[tiempo] presupuesto de {args.max_minutes:.0f} min agotado → guardando.", flush=True)
            stop["flag"] = True
            return True
        return False

    # El entrenamiento se envuelve en try/except para que CUALQUIER fallo
    # (OOM, etc.) no impida guardar y subir el mejor modelo logrado.
    try:
        # ===================== FASE A: Adam =====================
        if args.adam > 0:
            print(f"\n[fase A] Adam · {args.adam} épocas · lr={ent.adam_lr} · chunk={chunk}")
            opt = Adam(model.parameters(), lr=ent.adam_lr)
            t0 = time.time()
            for ep in range(args.adam):
                total, comps = compute_grads()
                opt.step()
                history.append({"step": ep, "loss": total, **comps})
                _save_if_best(total, comps)
                if ep % args.log_every == 0 or ep == args.adam - 1:
                    el = time.time() - t0
                    its = (ep + 1) / el
                    eta = (args.adam - ep - 1) / its
                    print(f"  [A] ep {ep:5d}/{args.adam} | loss={total:.3e} "
                          f"| res={comps['res']:.2e} load={comps['load']:.2e} "
                          f"bc={comps['bc']:.2e} ic={comps['ic']:.2e} "
                          f"| best={best['loss']:.2e} | {its:.1f} it/s | ETA {eta/60:.1f} min",
                          flush=True)
                if _should_stop():
                    break
            print(f"[fase A] completada en {time.time()-t0:.1f}s", flush=True)

        # ===================== FASE B: L-BFGS =====================
        if args.lbfgs > 0 and not stop["flag"]:
            print(f"\n[fase B] L-BFGS · {args.lbfgs} pasos · chunk={chunk}")
            opt = LBFGS(model.parameters(), lr=ent.lbfgs_lr,
                        max_iter=20, history_size=50, line_search_fn="strong_wolfe")
            t0 = time.time()
            hold = {"comps": {"res": 0.0, "load": 0.0, "bc": 0.0, "ic": 0.0, "eq": 0.0}}
            base_step = len(history)
            for ep in range(args.lbfgs):
                def closure():
                    total, comps = compute_grads()
                    hold["comps"] = comps
                    return torch.tensor(total, device=device)
                loss = opt.step(closure)
                lval = float(loss)
                history.append({"step": base_step + ep, "loss": lval, **hold["comps"]})
                _save_if_best(lval, hold["comps"])
                if ep % max(1, args.log_every // 5) == 0 or ep == args.lbfgs - 1:
                    el = time.time() - t0
                    print(f"  [B] paso {ep:4d}/{args.lbfgs} | loss={lval:.3e} "
                          f"| best={best['loss']:.2e} | {el/60:.1f} min", flush=True)
                if _should_stop():
                    break
            print(f"[fase B] completada en {time.time()-t0:.1f}s", flush=True)
    except Exception as exc:
        print(f"\n[error] entrenamiento interrumpido ({exc}); se guarda el mejor modelo hasta ahora.",
              flush=True)

    # --- Guardado (SIEMPRE se ejecuta: total, por tiempo o por interrupción) ---
    os.makedirs(os.path.dirname(ent.checkpoint), exist_ok=True)

    # Métricas de entrenamiento para la visualización (docs/03 §13.1): desglose
    # final de la pérdida y curva de convergencia (submuestreada a <=1000 puntos
    # para no inflar el checkpoint).
    final_comps = best.get("comps", {"res": float("nan"), "load": float("nan"),
                                      "bc": float("nan"), "ic": float("nan"),
                                      "eq": float("nan")})
    stride = max(1, len(history) // 1000)
    history_sub = history[::stride]
    payload = {
        "state_dict": best["state"] if best["state"] is not None else model.state_dict(),
        "best_loss": best["loss"],
        "model_cfg": vars(mod),
        "input_bounds": C.input_bounds(),
        "material_entrenamiento": material.nombre,
        "T_hat": dom.T_hat,
        "t_ramp_hat": C.VIGA.t_ramp_hat,
        # --- Métricas de entrenamiento (§13.1) ---
        "final_comps": final_comps,
        "loss_weights": {"res": ent.w_res, "load": ent.w_load, "bc": ent.w_bc,
                         "ic": ent.w_ic, "eq": ent.w_eq},
        "history": history_sub,
        "n_params": get_n_params(model),
        "adam_epochs": args.adam,
        "lbfgs_epochs": args.lbfgs,
        "n_collocation": {"res": ent.n_res, "bc": ent.n_bc, "ic": ent.n_ic},
    }
    try:
        torch.save(payload, ent.checkpoint)
        print(f"\n[ok] mejor pérdida = {best['loss']:.3e}")
        print(f"[ok] modelo guardado en {ent.checkpoint}")
    except Exception as exc:
        print(f"[error] no se pudo guardar el checkpoint local: {exc}", flush=True)

    # Subida a W&B (almacenamiento gratuito) para usar el modelo desde la app
    # local sin mantener encendido el Pod. CRÍTICO: si esto falla, el modelo se
    # pierde al apagar el Pod, así que se reintenta y se avisa MUY claramente.
    subido = False
    try:
        from utils.registry import upload_checkpoint
        subido = upload_checkpoint(ent.checkpoint, metadata={
            "best_loss": best["loss"],
            "material": material.nombre,
            "adam_epochs": args.adam, "lbfgs_epochs": args.lbfgs,
            "run_tag": os.environ.get("RUN_TAG", "manual"),
        })
    except Exception as exc:
        print(f"[wandb] ERROR en la subida: {exc}", flush=True)
    if subido:
        print("[wandb] ✅ modelo subido correctamente a W&B (alias 'latest').", flush=True)
    else:
        print("[wandb] ⚠️  el modelo NO se subió a W&B (revisa WANDB_API_KEY).", flush=True)
    print("[fin] entrenamiento terminado; el contenedor procederá a apagar el Pod.", flush=True)


if __name__ == "__main__":
    main()
