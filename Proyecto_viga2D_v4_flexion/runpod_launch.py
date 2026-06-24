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

⚠️  COSTE: usa GPU de ALTO RENDIMIENTO con un TECHO de 2.5 $/h (ver
   MAX_PRECIO_USD_H). Acelera mucho la autodiferenciación de 2.º orden; el
   entrenamiento completo (~3000 Adam + 500 L-BFGS) suele tardar < 25 min,
   con un coste típico < 1 $ aún en la GPU más cara.
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

# Techo de precio autorizado: no se crea ningún Pod cuyo coste on-demand lo
# supere (protección de créditos). Ajustable a voluntad.
MAX_PRECIO_USD_H = 2.5

# GPUs por preferencia: MÁS POTENTE primero (el usuario prioriza velocidad y
# autoriza hasta 2.5 $/h). La H100/A100 reducen el autograd de 2.º orden a unos
# pocos minutos; el resto son alternativas si la preferida no está disponible.
# El precio es orientativo (RunPod on-demand) y sólo se usa para no superar el
# techo MAX_PRECIO_USD_H.
GPU_PREFERENCES = [
    ("NVIDIA A100 80GB PCIe",       1.89),
    ("NVIDIA H100 PCIe",            2.39),
    ("NVIDIA L40S",                 1.03),
    ("NVIDIA RTX 6000 Ada",         0.99),
    ("NVIDIA RTX A6000",            0.79),
    ("NVIDIA GeForce RTX 4090",     0.69),
]

POLL_INTERVAL = 30      # s entre consultas a W&B

# ----------------------------------------------------------------------
# HIPERPARÁMETROS DEL ENTRENAMIENTO — FUENTE ÚNICA DE VERDAD.
# Se PASAN al Pod por variable de entorno (sobreescriben los ENV del Dockerfile),
# de modo que se cambian aquí / en .env SIN reconstruir la imagen Docker. Antes
# vivían sólo en el Dockerfile y había que reconstruir+publicar la imagen para
# cambiarlos (causa de líos: el reentreno usó 2000 épocas pese a editar el config).
# ----------------------------------------------------------------------
POD_ADAM = os.getenv("ADAM", "2000")
POD_LBFGS = os.getenv("LBFGS", "200")
POD_CHUNK = os.getenv("CHUNK", "1024")
POD_MAX_MINUTES = int(os.getenv("MAX_MINUTES", "120"))   # presupuesto de entreno del Pod

# CLAVE (bug que mataba el Pod antes de tiempo y perdía el modelo): el lanzador DEBE
# esperar SIEMPRE más que el presupuesto del Pod + el arranque (pull de la imagen, varios
# min) + la subida del modelo. El run de W&B sólo aparece AL FINAL, así que si
# POLL_TIMEOUT < MAX_MINUTES el launcher agota su espera con el entreno en curso y, en su
# `finally`, TERMINA el Pod → entrenamiento abortado y nada subido a W&B. Por eso se
# DERIVA de POD_MAX_MINUTES y nunca puede quedar por debajo.
POLL_TIMEOUT = (POD_MAX_MINUTES + 10) * 60     # presupuesto + 10 min (arranque + subida)


def main():
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    image = os.environ.get("DOCKER_IMAGE", "miusuario/viga2d-train:latest")
    run_tag = wandb.util.generate_id()
    print(f"[runpod] run_tag = {run_tag}")
    print(f"[runpod] entreno: Adam={POD_ADAM}, L-BFGS={POD_LBFGS}, chunk={POD_CHUNK}, "
          f"presupuesto Pod={POD_MAX_MINUTES} min")
    print(f"[runpod] el lanzador esperará hasta {POLL_TIMEOUT // 60} min "
          f"(presupuesto + 10 de margen) antes de darse por vencido.")

    pod = None
    for gpu, precio in GPU_PREFERENCES:
        if precio > MAX_PRECIO_USD_H:
            print(f"[runpod] {gpu} (~{precio} $/h) supera el techo "
                  f"{MAX_PRECIO_USD_H} $/h; se omite.")
            continue
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
                    # CLAVE: permite que el contenedor APAGUE SU PROPIO POD al
                    # terminar (runpod_selfterminate.py), sin depender de este
                    # lanzador. Evita reinicios en bucle y consumo de créditos.
                    "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
                    # Hiperparámetros: el lanzador es la fuente única de verdad y
                    # sobreescribe los ENV del Dockerfile (sin reconstruir la imagen).
                    "ADAM": POD_ADAM,
                    "LBFGS": POD_LBFGS,
                    "CHUNK": POD_CHUNK,
                    "MAX_MINUTES": str(POD_MAX_MINUTES),
                },
            )
            print(f"[runpod] Pod creado en {gpu} (~{precio} $/h): {pod['id']}")
            break
        except Exception as exc:
            print(f"[runpod] {gpu} no disponible ({exc}); probando siguiente...")

    if pod is None:
        raise SystemExit("[runpod] No se pudo crear el Pod con ninguna GPU dentro del presupuesto.")

    try:
        _wait_for_wandb(run_tag)
    finally:
        # El contenedor se autoapaga al terminar (runpod_selfterminate.py), así que
        # aquí el Pod puede estar YA terminado: "pod not found" es ÉXITO, no error.
        try:
            runpod.terminate_pod(pod["id"])
            print(f"[runpod] Pod {pod['id']} TERMINADO (créditos protegidos).")
        except Exception as exc:
            if "not found" in str(exc).lower():
                print(f"[runpod] Pod {pod['id']} ya estaba apagado (autoapagado del "
                      "contenedor). Créditos protegidos. ✅")
            else:
                print(f"[runpod] ⚠️  no se pudo terminar el Pod {pod['id']}: {exc}. "
                      "Verifícalo en el panel de RunPod.")


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
