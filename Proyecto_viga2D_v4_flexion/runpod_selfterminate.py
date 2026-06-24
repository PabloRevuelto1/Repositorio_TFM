"""
runpod_selfterminate.py
=======================
Apaga el propio Pod de RunPod desde DENTRO del contenedor, en cuanto el
entrenamiento ha terminado (con éxito, por límite de tiempo o por error).

Es la salvaguarda CLAVE contra el consumo descontrolado de créditos: no depende
del proceso lanzador (`runpod_launch.py`), que podría haberse caído o perder la
conexión. RunPod inyecta el identificador del Pod en la variable de entorno
`RUNPOD_POD_ID`; con `RUNPOD_API_KEY` (que el lanzador inyecta en el Pod) se
llama a la API para terminarlo.

Se invoca SIEMPRE al final del CMD del contenedor (`... ; python runpod_selfterminate.py`),
de modo que el Pod se destruye pase lo que pase y NO se reinicia el entrenamiento.
Degrada con elegancia: si falta algo, solo avisa (no relanza nada).
"""

import os
import sys


def main():
    pod_id = os.getenv("RUNPOD_POD_ID")
    api_key = os.getenv("RUNPOD_API_KEY")

    if not pod_id:
        print("[selfterminate] RUNPOD_POD_ID no definido (¿fuera de RunPod?); no se apaga nada.")
        return
    if not api_key:
        print("[selfterminate] ⚠️  RUNPOD_API_KEY no definido en el Pod: NO puedo apagarme "
              "solo. Termínalo manualmente desde el panel de RunPod para no gastar créditos.")
        return

    try:
        import runpod
    except ImportError:
        print("[selfterminate] ⚠️  paquete 'runpod' no instalado; no puedo autoapagarme.")
        return

    runpod.api_key = api_key
    try:
        runpod.terminate_pod(pod_id)
        print(f"[selfterminate] ✅ Pod {pod_id} TERMINADO desde el contenedor "
              "(no se acumulan más créditos).")
    except Exception as exc:
        print(f"[selfterminate] ⚠️  no se pudo terminar el Pod {pod_id}: {exc}. "
              "Termínalo manualmente desde RunPod.", file=sys.stderr)


if __name__ == "__main__":
    main()
