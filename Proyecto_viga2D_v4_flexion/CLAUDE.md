# CLAUDE.md — Proyecto Viga 2D · PINNsFormer (TFM)

> Este fichero se carga automáticamente cada sesión. Es la **memoria viva** del
> proyecto: léelo antes de actuar y **mantenlo actualizado**. El detalle profundo
> está en `docs/` y en las skills de `.claude/skills/` (se cargan bajo demanda).

## Qué es el proyecto
PINN para la **elastodinámica 2D de una viga empotrada** (tensión plana,
Navier-Cauchy) bajo carga cortante de impacto, resuelta con la arquitectura
**PINNsFormer** (Zhao et al., ICLR 2024) y **validada contra un ground truth por
Elementos Finitos** (scikit-fem). Es un TFM; el usuario lo defenderá ante tribunal.

**Versión actual = SIMPLIFICADA** (feedback de tutores: "impresionante pero
demasiado complejo"):
- **Escenario fijo** (`config.VIGA`): L=1.5 m, c=0.10 m, P_max=100 kN/m, acero único.
- Red **d_in=3** (ξ, η, t̂) → d_out=2 (U, V). Antes era paramétrica (d_in=7).
- Carga en **rampa lineal** `g(t̂)=min(t̂/t̂_ramp,1)`, `t̂_ramp=20` (≈0.3 periodos de flexión).
- **Restricciones duras**: la salida se multiplica por `ξ · env(t̂)`. `ξ` → cero en
  empotramiento. `env` según `hard_ic`: `t̂²/(t̂²+τ²)` (CI dura: u y u̇ nulas) o, por
  defecto **`hard_ic=False`**, `t̂/(t̂+τ)` (CI de DESPLAZAMIENTO dura, VELOCIDAD blanda →
  `u̇(0)=0` vía `w_ic` para que emerja la oscilación). Ambas saturan a 1.
- **Currículo temporal** (`curriculum_stages`): Adam por etapas con ventana `[0,t_frac·T̂]`
  creciente → respeta la causalidad (como el FEM) y deja emerger la oscilación.
- **Pesos causales** (`config.causal`, `pde_loss._causal_residual`; Wang et al. 2022): el
  residuo se pondera por franjas con `w_i=exp(-ε·Σ_{j<i}L_j)` → fuerza a resolver el tiempo en
  ORDEN, propagando el reposo inicial (emula el marchado del FEM). Es el fix del NO-REBOTE
  (docs/05 §4-septies): la oscilación es vibración libre con señal `~ω̂²≈0.018` bajo el suelo de
  residuo; sólo la causalidad la separa de la cuasi-estática. **Re-escalar t̂ NO sirve (probado):
  ω̂² es un adimensional físico invariante a la unidad de tiempo.** Reserva: seq2seq.

**RÉGIMEN DE FLEXIÓN (2026-06-22, docs/05) — el cambio de fondo:** el caso estaba
planteado en el régimen de ONDAS (T_hat=4 → 0.77 ms = 5.7 % de un periodo de flexión)
mientras se validaba contra teoría de flexión (Euler-Bernoulli) → resultados ~100×
blandos y checks inválidos. Corregido: `T_hat=100` (≈1.44 periodos de flexión),
`SIGMA_SCALE` = tensión de flexión `P L²/(2c³)` (antes la tracción `3P/4c`) → flecha
adimensional O(1) (antes ~225), ansatz saturante, residuo reescalado `×ĉ` (antes `×ĉ²`,
que infra-imponía la EDP). **Pendiente reentrenar en RunPod.** Termómetro = **error L2
PINN vs FEM**, no la pérdida total.

## Mapa de ficheros
- `config.py` — escalas, material, escenario `VIGA`, hiperparámetros, `VALIDACION`.
- `models/pinnsformer.py` — arquitectura del paper (NO TOCAR las capas) + ansatz de restricción dura.
- `physics/pde_loss.py` — `ElastodynamicsLoss`: residuo (×ĉ), carga, contorno, IC, equilibrio seccional.
- `fem/fem_solver.py` — ground truth FEM (Newmark, P1, malla = rejilla de inferencia).
- `utils/` — `sampling.py` (LHS+pseudo-secuencia), `inference.py`, `metrics.py` (sim + comparación PINN/FEM), `registry.py` (W&B).
- `train.py` — Adam→L-BFGS; guarda checkpoint + sube a W&B.
- `app.py` — dashboard Dash: PINN vs FEM (gráfico + tabla de ganador).
- `runpod_launch.py` + `Dockerfile.train` + `runpod_selfterminate.py` — entreno cloud.
- `docs/00..04` — documentación didáctica de defensa. `docs/BITACORA.md` — historial.

## Pesos de pérdida actuales
`w_res=1, w_load=20, w_bc=5, w_ic=10, w_eq=5` (`L_eq` PLENO en todo t̂ — la relajación temporal
de la Fase 10 fue un error que regresionó la estática, revertida en Fase 11 / docs/05 §4-sexies;
`w_ic` alto porque ahora impone la velocidad inicial, no el ansatz). **`w_eq` = EQUILIBRIO SECCIONAL**
(`pde_loss.loss_equilibrium`): impone el cortante/momento integral por sección
(`∫τ̂_xy dη=-0.02·g`, `∫σ̂_xx·η dη=+0.3·(1-ξ)·g`) que rompe el COLAPSO TRIVIAL del
problema controlado por fuerza (EDP homogénea → `u≈0` da residuo≈0; la pérdida total no
lo detecta). Física pura (consecuencia de ∇·σ=0, verificada vs FEM). Blando: si la
flecha sale corta subir `w_eq`; si sale demasiado cuasi-estática (sin oscilar), bajarlo.
Ver docs/05 §4-ter. El balance de pérdidas sigue siendo el primer knob de ajuste.

## REGLAS DE TRABAJO (cúmplelas siempre)
1. **No modificar la arquitectura PINNsFormer del paper** (`models/pinnsformer.py`,
   capas WaveAct/Encoder/Decoder/atención). Todo lo demás es hiperparámetro libre:
   pesos, puntos de colocación, ansatz de salida, escenario, etc.
2. **Economía de tokens**: respuestas y diffs concisos; lee solo lo necesario; no
   re-derives lo ya establecido; no re-leas ficheros recién editados.
3. **SOLID y código limpio**: una responsabilidad por módulo/función; nombres y
   estilo coherentes con el código existente (español en docstrings/comentarios,
   comentarios densos explicando el *porqué* físico/numérico, no el *qué*).
4. **No ejecutar tests pesados en WSL/CPU**: la autodiferenciación de **2.º orden**
   sobre miles de puntos agota la RAM y tumba WSL. Valida con `py_compile` y lotes
   diminutos (≤16 pts); el entrenamiento real va en **RunPod GPU**.
5. **Seguridad de costes en RunPod**: el pod debe **autoapagarse** pase lo que pase
   (`runpod_selfterminate.py` encadenado en el CMD); el modelo debe subirse a W&B
   con reintentos. Nunca dejar un pod que pueda reiniciar el entreno en bucle.
6. **Documentación sincronizada**: cualquier cambio de comportamiento se refleja en
   el `README.md`, en los `docs/` afectados y en `docs/BITACORA.md`.
7. **Coherencia PINN↔FEM**: el perfil de carga `g(t̂)` y la geometría deben ser
   IDÉNTICOS en `physics/pde_loss.py` y `fem/fem_solver.py` (comparación justa).
8. **Adimensionalización**: trabajar siempre en variables adimensionales; en SI la
   PINN no converge (residuo escalado ~1e11).

## Flujo de trabajo típico
1. Tocar código → `python -m py_compile <ficheros>` (rápido, seguro).
2. Validar lógica con lotes diminutos en CPU (sin entrenar).
3. Entrenar en RunPod (`runpod_launch.py`); el modelo va a W&B (`:latest`).
4. Visualizar/comparar en local: `python app.py` (descarga de W&B si el local falla).
5. **Actualizar `docs/BITACORA.md`** con el cambio.

## Skills disponibles (`.claude/skills/`)
- `pinn-debugging` — diagnosticar/arreglar PINNs (colapso trivial, condicionamiento, hard constraints).
- `pinnsformer-arch` — detalles de la arquitectura y qué se puede/no tocar.
- `scikit-fem-elasto` — patrones del solver FEM de referencia (elastodinámica, Newmark).
- `runpod-mlops` — entreno cloud seguro en costes + W&B + app.
