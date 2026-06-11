"""
runpod_launch.py
================
Lanzador de ENTRENAMIENTO en RunPod (sin dependencias de orquestación pesadas).

Crea un Pod GPU on-demand a partir de la imagen `Dockerfile.train`, espera a
que el run de entrenamiento aparezca como `finished` en W&B y TERMINA el Pod
automáticamente para no malgastar créditos.

Requisitos (variables en `.env`):
    RUNPOD_API_KEY   clave de RunPod
    WANDB_API_KEY    clave de W&B (se inyecta en el Pod)
    DOCKER_IMAGE     imagen publicada, p. ej. "miusuario/viga2d-train:latest"

Uso:
    python runpod_launch.py

⚠️  COSTE: usa GPU económica (RTX A4000 ≈ 0.20-0.34 $/h). El entrenamiento
   completo (~5000 Adam + 200 L-BFGS) suele tardar < 1 h → < 0.5 $.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

import wandb

try:
    import runpod
except ImportError:
    raise SystemExit("Instala el SDK:  pip install runpod")

# GPUs por preferencia: mejor relación precio/rendimiento primero. La RTX 4090
# (24 GB) es mucho más rápida en autograd de 2.º orden y suele entrenar en
# < 45 min (≈ 0.4 $). Todas se mantienen por debajo de ~0.7 $/h → < 1 $/entreno.
GPU_PREFERENCES = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A4000",
]

POLL_INTERVAL = 30      # s entre consultas a W&B
# Margen: mayor que MAX_MINUTES del Pod (70 min) para que dé tiempo a guardar
# y subir el modelo antes de que el launcher termine el Pod.
POLL_TIMEOUT = 5400     # s (90 min)


def main():
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    image = os.environ.get("DOCKER_IMAGE", "miusuario/viga2d-train:latest")
    run_tag = wandb.util.generate_id()
    print(f"[runpod] run_tag = {run_tag}")

    pod = None
    for gpu in GPU_PREFERENCES:
        try:
            pod = runpod.create_pod(
                name=f"viga2d-train-{run_tag}",
                image_name=image,
                gpu_type_id=gpu,
                gpu_count=1,
                container_disk_in_gb=20,
                env={
                    "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
                    "WANDB_PROJECT": os.getenv("WANDB_PROJECT", "viga2d-pinnsformer"),
                    "RUN_TAG": run_tag,
                },
            )
            print(f"[runpod] Pod creado en {gpu}: {pod['id']}")
            break
        except Exception as exc:
            print(f"[runpod] {gpu} no disponible ({exc}); probando siguiente...")

    if pod is None:
        raise SystemExit("[runpod] No se pudo crear el Pod con ninguna GPU preferida.")

    try:
        _wait_for_wandb(run_tag)
    finally:
        runpod.terminate_pod(pod["id"])
        print(f"[runpod] Pod {pod['id']} TERMINADO (créditos protegidos).")


def _wait_for_wandb(run_tag):
    """Espera a que el run aparezca como finished/failed en W&B."""
    wandb.login(key=os.environ["WANDB_API_KEY"])
    project = os.getenv("WANDB_PROJECT", "viga2d-pinnsformer")
    entity = wandb.Api().default_entity
    path = f"{entity}/{project}"
    deadline = time.time() + POLL_TIMEOUT
    print(f"[runpod] esperando a que finalice el entrenamiento en {path}...")
    while time.time() < deadline:
        try:
            runs = list(wandb.Api().runs(path, filters={"config.run_tag": run_tag}))
        except ValueError:
            # El proyecto/run todavía no existe (el Pod aún entrena: el run de
            # W&B sólo se crea al subir el modelo al final). Seguir esperando.
            runs = []
        for r in runs:
            if r.state == "finished":
                print(f"[runpod] entrenamiento finalizado: {dict(r.summary)}")
                return
            if r.state in ("failed", "crashed"):
                raise RuntimeError(f"Entrenamiento {r.state}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError("El entrenamiento no finalizó dentro del tiempo límite.")


if __name__ == "__main__":
    main()
