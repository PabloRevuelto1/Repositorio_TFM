---
name: pinnsformer-arch
description: Detalles de la arquitectura PINNsFormer (Zhao et al., ICLR 2024) tal como se usa en el proyecto viga2D. Úsala al tocar models/pinnsformer.py, la pseudo-secuencia temporal, la activación WaveAct, o al decidir qué se puede y qué NO se puede modificar de la red. Cubre la regla de oro (no alterar las capas del paper) y cómo extender por la entrada/salida.
---

# Arquitectura PINNsFormer

## Regla de oro
**NO modificar las capas del paper** en `models/pinnsformer.py`: WaveAct,
FeedForward, EncoderLayer, DecoderLayer, Encoder, Decoder y el esquema residual
(`x = x + Attn(x)`, `x = x + FF(x)`). Es fidelidad al repo oficial (TFM/pinnsformer).
Todo lo demás —embedding de entrada, cabeza de salida, ansatz de restricción dura,
normalización— es adaptación legítima al problema.

## Componentes (fieles al paper)
- **WaveAct**: `f(x) = w1·sin(x) + w2·cos(x)`, con w1,w2 aprendibles. Activación
  periódica: capta mejor oscilaciones de problemas dinámicos que tanh/ReLU.
- **Encoder–Decoder Transformer**: self-attention en encoder, cross-attention en
  decoder, conexiones residuales. Hiperparámetros del paper: `d_model=32, heads=2, N=1`.
- **Pseudo-secuencia temporal** (`utils/sampling.make_time_sequence`): cada punto
  `(ξ,η,t̂)` se replica `k=5` veces desplazando SOLO t̂ un paso `seq_step=1e-3`. El
  Transformer procesa esa mini-secuencia; la predicción "real" es el **primer token**
  (índice 0). Convierte un problema puntual en secuencial (donde el Transformer brilla).

## Adaptaciones de ESTE proyecto (permitidas)
- **Entrada d_in=3** (ξ,η,t̂): escenario fijo, sin parámetros. La versión paramétrica
  usaba d_in=7 (+ L̂,ĉ,p̂,t̂_imp); recuperable cambiando el embedding y el muestreo.
- **Salida d_out=2** (U,V), vía cabeza MLP `32→512→512→2` con WaveAct.
- **Normalización de entrada** a ~[-1,1] con buffers `(in_center, in_halfspan)`.
- **Restricción dura** en `forward`: tras la cabeza, `output *= ξ·(t̂/T̂)²`. Es una
  transformación de la salida (hard-constraint BC), NO una capa del paper. La
  autodiferenciación atraviesa el factor → residuo y tensiones se calculan sobre la
  solución ya restringida; train e inferencia quedan automáticamente consistentes.

## Entradas/derivadas (autograd)
- El forward recibe `(xi, eta, t)` como tensores SEPARADOS con `requires_grad=True`
  para poder derivar el residuo respecto a las coordenadas.
- 1.er orden (create_graph=False) → inferencia (tensiones). 2.º orden
  (create_graph=True) → residuo de la EDP (operación pesada, solo GPU).

## Persistencia del modelo
- El checkpoint guarda `model_cfg = vars(ModeloCfg)` (incluye `d_in`, `hard_constraints`)
  + `input_bounds` + `T_hat`. `inference.load_model` reconstruye con esos valores y
  **rechaza** checkpoints con `d_in` distinto del actual (incompatibles).
