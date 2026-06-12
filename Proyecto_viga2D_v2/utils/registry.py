"""
utils/registry.py
=================
Registro del modelo entrenado en Weights & Biases (W&B).

Motivación (optimización de créditos RunPod):
  * El entrenamiento se ejecuta en un Pod GPU de RunPod (la diferenciación
    automática de segundo orden es inviable en CPU / WSL).
  * El checkpoint resultante (`best_model.pth`, unos pocos MB) se sube como
    *artifact* a W&B, cuyo almacenamiento es GRATUITO en el plan personal.
  * Así el Pod de RunPod puede apagarse inmediatamente tras entrenar (no se
    paga almacenamiento) y la app de visualización descarga el modelo desde
    W&B para ejecutarse en local, sin consumir créditos.

Uso totalmente OPCIONAL: si no hay WANDB_API_KEY o `wandb` no está instalado,
las funciones degradan con elegancia y el proyecto sigue funcionando con el
checkpoint local.
"""

import os

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "viga2d-pinnsformer")
ARTIFACT_NAME = os.getenv("WANDB_ARTIFACT", "pinnsformer-viga2d")


def _wandb_available():
    if not os.getenv("WANDB_API_KEY"):
        return None
    try:
        import wandb
        return wandb
    except ImportError:
        return None


def upload_checkpoint(local_path, metadata=None):
    """Sube `local_path` como artifact de modelo a W&B. Devuelve True si tuvo éxito."""
    wandb = _wandb_available()
    if wandb is None:
        print("[wandb] omitido (sin WANDB_API_KEY o sin paquete wandb).")
        return False
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    with wandb.init(project=WANDB_PROJECT, job_type="train",
                    config=metadata or {}) as run:
        art = wandb.Artifact(ARTIFACT_NAME, type="model",
                             metadata=metadata or {})
        art.add_file(local_path)
        run.log_artifact(art)
    print(f"[wandb] checkpoint subido como artifact '{ARTIFACT_NAME}'.")
    return True


def download_checkpoint(dest_dir="checkpoints", alias="latest"):
    """
    Descarga el artifact del modelo desde W&B a `dest_dir`.
    Devuelve la ruta local al .pth o None si no es posible.
    """
    wandb = _wandb_available()
    if wandb is None:
        return None
    try:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        api = wandb.Api()
        art = api.artifact(f"{WANDB_PROJECT}/{ARTIFACT_NAME}:{alias}")
        path = art.download(root=dest_dir)
        for f in os.listdir(path):
            if f.endswith(".pth"):
                return os.path.join(path, f)
    except Exception as exc:  # red, credenciales, artifact inexistente...
        print(f"[wandb] no se pudo descargar el modelo: {exc}")
    return None
