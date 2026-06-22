---
name: runpod-mlops
description: Flujo MLOps seguro en costes para entrenar la PINN en RunPod, subir a Weights & Biases y servir/visualizar en local. Úsala al tocar train.py, runpod_launch.py, Dockerfile.train, runpod_selfterminate.py, utils/registry.py o app.py (carga del modelo). Cubre el autoapagado del pod (anti consumo de créditos), la subida robusta a W&B y la carga del modelo en la app.
---

# MLOps RunPod + W&B + app local (seguridad de costes)

Reparto: entreno (2.º orden, pesado) en **RunPod GPU**; modelo en **W&B** (gratis);
visualización (1.er orden, ligera) en **local/WSL**. La app NO reentrena.

## REGLA #1 — el pod debe autoapagarse pase lo que pase
El bug que vació créditos: `train.py` petaba antes de guardar → el contenedor salía
con error → RunPod **reiniciaba el entreno en bucle**. Defensa en capas:
1. **Autoapagado desde el contenedor** (`runpod_selfterminate.py`): usa `RUNPOD_POD_ID`
   (lo inyecta RunPod) + `RUNPOD_API_KEY` para `runpod.terminate_pod(...)`.
2. **CMD encadenado** en `Dockerfile.train`: `python train.py ... ; python runpod_selfterminate.py`.
   El `;` garantiza que el apagado corre AUNQUE train.py falle → nunca bucle.
3. `runpod_launch.py` inyecta `RUNPOD_API_KEY` en el pod y trata `"pod not found"`
   al terminar como **éxito** (el contenedor ya se apagó).
4. **Presupuesto de tiempo** (`--max-minutes`, env `MAX_MINUTES`): al superarlo,
   `train.py` para tras la época en curso (handler de SIGTERM + chequeo `_should_stop`)
   y guarda el MEJOR estado, no el último.

## REGLA #2 — la subida a W&B no puede perderse
`utils.registry.upload_checkpoint`: login + `wandb.init` → `Artifact` → `log_artifact(
art, aliases=["latest"])` → `art.wait()` (bloquea hasta subir). **3 reintentos** y
logs claros (`✅ subido` / `⚠️ NO subido`). El guardado local y la subida van envueltos
en try/except para no impedir la salida limpia + autoapagado.

## REGLA #3 — la app debe encontrar el modelo aunque el local sea viejo
`app._get_model`: intenta el checkpoint LOCAL; si no existe o es **incompatible**
(p. ej. d_in antiguo), **descarga de W&B (`:latest`) y carga ese**. Nunca dejar que
un checkpoint local obsoleto "esconda" al bueno de W&B. `load_model` rechaza
`d_in≠` actual con mensaje claro.

## Checklist al lanzar un entreno
1. `.env` con `RUNPOD_API_KEY`, `WANDB_API_KEY`, `DOCKER_IMAGE` (y opcional `WANDB_PROJECT`).
2. **Reconstruir y publicar la imagen** si cambió el código:
   `docker build -f Dockerfile.train -t <user>/viga2d-train:latest . && docker push ...`.
3. `python runpod_launch.py` (GPU por preferencia, techo `MAX_PRECIO_USD_H=2.5`).
4. En los logs del pod, al final, confirmar: `[wandb] ✅ modelo subido` y
   `[selfterminate] ✅ Pod ... TERMINADO`.
5. `python app.py` en local → descarga `:latest` si el local falla.

## Errores y su lectura
- `pod not found to terminate` en el launcher = **éxito** (autoapagado del contenedor).
- App "sin modelo" pero W&B tiene artifact = checkpoint local incompatible escondía
  al de W&B → ya resuelto por el fallback de `_get_model`.
- Modelo NO en W&B = train.py petó antes de subir, o se mató el contenedor antes de
  `art.wait()` → revisar logs; bajar `MAX_MINUTES` si no daba tiempo.

## Ahorro
- Baja `MAX_MINUTES` / épocas (Dockerfile env `ADAM`,`LBFGS`) si el entreno cabe en
  menos tiempo. El mejor modelo se guarda igual al cortar por tiempo.
