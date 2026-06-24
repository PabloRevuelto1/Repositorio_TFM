# 📓 Bitácora del proyecto — Viga 2D · PINNsFormer

Historial cronológico y **memoria persistente** del proyecto. Claude debe
**añadir una entrada nueva con cada cambio significativo** (formato: fecha, qué,
por qué, ficheros). Lo más reciente arriba. Para el estado *actual* resumido,
ver [`../CLAUDE.md`](../CLAUDE.md).

---

## Estado actual (resumen)
- **Escenario fijo** (`config.VIGA`): L=1.5 m, c=0.10 m, P_max=100 kN/m, acero.
- **PINNsFormer** d_in=3 (ξ,η,t̂), d_out=2, con **restricción dura** `ξ·(t̂/T̂)²`.
- **Carga**: rampa lineal `g(t̂)=min(t̂/t̂_ramp,1)`.
- **Pérdida**: `w_res=1, w_load=20, w_bc=5, w_ic=1`; residuo reescalado ×ĉ².
- **Ground truth FEM** (`fem/fem_solver.py`, scikit-fem + Newmark) integrado en la app.
- **MLOps**: RunPod (autoapagado) → W&B (`:latest`) → app local con comparación PINN/FEM.
- Último checkpoint en W&B: `best_loss ≈ 2.0e-3` (entrenó sin colapsar).

---

## Cronología

### Fase 6 — Seguridad de costes en RunPod y carga del modelo
- **Bug crítico (consumo de créditos)**: tras simplificar, `train.py` seguía usando
  `dom.alpha_ramp` (atributo eliminado) en el `payload`, fuera del try/except → al
  acabar/cortarse el entreno petaba con `AttributeError` antes de guardar/subir →
  RunPod reiniciaba el contenedor → reentreno en bucle, gastando créditos. **Fix**:
  `dom.alpha_ramp`→`C.VIGA.t_ramp_hat`; guardado/subida envueltos y con logs claros.
- **Autoapagado del pod** (`runpod_selfterminate.py`, nuevo): el contenedor mata su
  propio Pod (`RUNPOD_POD_ID`+`RUNPOD_API_KEY`) al terminar el CMD (`train.py ; selfterminate`),
  pase lo que pase → sin reinicios en bucle. `runpod_launch.py` inyecta `RUNPOD_API_KEY`
  y trata "pod not found" como éxito (ya autoapagado).
- **Subida a W&B robusta** (`utils/registry.py`): 3 reintentos, alias `latest`, `art.wait()`.
- **App no cargaba el modelo**: un checkpoint local antiguo (d_in=7) "escondía" al de
  W&B porque solo se descargaba si NO existía fichero local. **Fix** (`app.py`,
  `_get_model`): intenta local; si falla (incompatible), **descarga de W&B y carga eso**.
  `load_model` rechaza checkpoints con `d_in≠3` con mensaje claro.

### Fase 12 — PESOS CAUSALES (fix del no-rebote); el escalado temporal descartado (2026-06-23) → [05 §4-septies](05_Diagnostico_resultados.md)
- **6.º reentreno (L_eq pleno, w_eq=5)**: estática YA correcta (flecha 703µm, EB 803 → 12.5% ✓)
  pero **sigue sin rebotar** (L2 ~52-54%, σ_VM ~24 vs 48 MPa, "periodo"=ventana = artefacto).
- **Auditoría**: el punto 3 (re-escalar t̂ por el periodo de flexión) **NO sirve** — demostrado:
  la señal de oscilación `ω̂²≈0.018` es un adimensional físico (esbeltez²) invariante a la unidad
  de tiempo. Descartado para no malgastar cómputo.
- **Causa real (contrastada)**: modo de fallo de PINN globales-en-tiempo en transitorios
  oscilatorios (Krishnapriyan 2021, Wang 2022). La cuasi-estática y la dinámica tienen residuo
  casi idéntico (Δ~ω̂², bajo el suelo RMS≈0.017); la global elige la cuasi-estática (que viola
  `u̇(0)=0`). El FEM gana por **marchar en el tiempo**.
- **Fix**: **pesos causales** (`config.causal=True`, `eps_causal=1.0`, `n_causal_bins=32`;
  `pde_loss._causal_residual`). `w_i=exp(-ε·Σ_{j<i}L_j)` fuerza a resolver el residuo en orden
  temporal → propaga el reposo inicial (emula el marchado del FEM). Binning validado en CPU.
- **Rectificación**: en Fase 10 desestimé los pesos causales por un argumento incompleto; la
  solución dinámica ES la causal, así que son justo lo que la selecciona. Reserva: seq2seq.

### Fase 11 — Reversión de la relajación de L_eq (regresión) + diagnóstico del no-rebote (2026-06-23) → [05 §4-sexies](05_Diagnostico_resultados.md)
- **5.º reentreno (causalidad, w_eq=3 + relajación de L_eq)**: REGRESIÓN. Flecha estática
  595µm (vs 833 con w_eq=5; EB 803), σ_VM máx. **al centro** (no empotramiento) y a ~40 %
  (19 vs 48 MPa FEM), sin rebote. L2 ~48-53 %.
- **Causa**: la relajación temporal `exp(-(t̂-t_ramp)/τ)` que metí en Fase 10 **apaga L_eq
  en el 80 % de la ventana**. L_eq era el ÚNICO término que fija la FORMA del momento
  (`M̂∝(1-ξ)`, máx. en empotramiento) → al apagarlo, la distribución de tensión se degrada.
- **Fix**: revertida la relajación (`pde_loss.loss_equilibrium` pleno en todo t̂); `w_eq` 3→5.
  El equilibrio es física EXACTA a todo instante (∇·σ=0), no sólo en la rampa.
- **No-rebote (causa profunda)**: con tiempo en unidades de ONDA (T_REF), la señal inercial
  que separa oscila/cuasi-estático es `~ω̂²≈8e-3`, MENOR que el suelo de residuo alcanzado
  (`L_res=3e-4 → RMS≈1.7e-2`) → el optimizador no la "ve". Agravado por **L-BFGS truncado
  a 40/100 épocas** (la fase de precisión que bajaría ese suelo). Pesos causales NO ayudan
  (no es desbalance de términos). Reserva si persiste: re-escalar t̂ por el periodo de flexión.

### Fase 10 — CAUSALIDAD: CI de velocidad blanda + relajación de L_eq + currículo temporal (2026-06-23) → [05 §4-quinquies](05_Diagnostico_resultados.md)
4.º reentreno (w_eq=5): flexa bien (833µm, EB 12.7%) pero NO oscila; L2 vs FEM estancado
~52%. Causa de fondo: la oscilación es la **vibración libre** (modo propio) y un modo libre
tiene **residuo EDP=0** → la pérdida no distingue cuasi-estática de dinámica; ambas son
mínimos y gana la "simple". No es épocas ni pesos (los adaptativos NO lo curan). El FEM
oscila porque marcha en el tiempo (causalidad). **Fix = inyectar causalidad** (estrategias
(3)+(2) que pidió el usuario): (3a) **CI de velocidad BLANDA** — ansatz `ξ·t̂²/(t̂²+τ²)` →
`ξ·t̂/(t̂+τ)` (`MODELO.hard_ic=False`): anula sólo el desplazamiento, `u̇(0)=0` pasa a pérdida
(`w_ic` 1→10) → excita el modo libre; (3b) **L_eq menor + relajación temporal** (`w_eq` 5→3,
`eq_relax_tau=20`: `exp(-(t̂-t_ramp)/τ)` desvanece el constraint cuasi-estático tras la
rampa); (2) **currículo temporal** (`curriculum_stages=((.3,.3),(.6,.3),(1,.4))`, `train.py`
en etapas con ventana `[0,t_frac·T̂]` creciente; `sampling.build_training_sets(t_max=)`; el
mejor modelo se decide en la ventana completa). Validado py_compile + lote diminuto (u(0)=0
exacto, u̇(0)≠0; recorte t_max; relax OK). Pesos adaptativos NO implementados (no atacan esto).
**Pendiente reentrenar.** Si no basta → pesos causales (Wang et al.). Tocados: config,
pinnsformer, pde_loss, sampling, train, inference; docs 02/04/05 + CLAUDE + skill pinn-debugging.

### Fase 9 — BUG del lanzador RunPod: mataba el Pod antes de tiempo (2026-06-23)
Dos entrenamientos perdidos (créditos gastados, modelo no subido a W&B). Causa: en
`runpod_launch.py`, `POLL_TIMEOUT=5400s (90 min)` era **menor** que `MAX_MINUTES=120`
del Pod (el usuario lo subió). El run de W&B sólo aparece al final (al subir), así que el
launcher agotaba su espera con el entreno en curso y, en su `finally`, **terminaba el Pod**
→ entreno abortado, nada en W&B. **Fix (fuente única de verdad):** el lanzador define
`POD_ADAM/LBFGS/CHUNK/MAX_MINUTES` (desde `.env`/shell, defaults 6000/400/1024/120), los
**pasa al Pod por env** (sobreescriben el Dockerfile → cambiar épocas YA NO requiere
reconstruir la imagen) y **deriva** `POLL_TIMEOUT=(MAX_MINUTES+30)·60` (invariante
`POLL_TIMEOUT>MAX_MINUTES`). Imprime épocas+timeout al arrancar. Tocados: runpod_launch.py,
Dockerfile.train (nota), skill runpod-mlops (REGLA #1-bis). Validado py_compile + invariante.

### Fase 8 — 3.er reentreno: flexa pero NO oscila (cuasi-estático) (2026-06-22) → [05 §4-quater](05_Diagnostico_resultados.md)
El equilibrio seccional rompió el colapso: flecha 843µm (EB 803, err 12.9%), σ_VM 27MPa
cerca del empotramiento, **L2 vs FEM 99%→51%**. Pero la viga NO rebota (cuasi-estática);
la "frecuencia" sigue siendo artefacto de ventana. La PINN convergió a la solución
**cuasi-estática**, no a la dinámica (el rebote del FEM es la vibración libre del oscilador
viga: inercia → sobreoscila ~2× y vuelve, a ~74Hz; correcto y esperado). Causas del 51%
(no es muro del método, la física de flexión ya está): (1) **w_eq=20 demasiado alto** y su
target es cuasi-estático (M̂=K(1-ξ)g, constante tras la rampa → prohíbe oscilar) → bajado a
**5**; (2) **infra-entreno 2000/200**: el `Dockerfile.train` fijaba `ENV ADAM=2000` que
anulaba el config → subido a **ADAM=6000, LBFGS=400, MAX_MINUTES=90**; (3) falta
**entrenamiento causal** (PINN vanilla converge al promedio temporal = cuasi-estática;
fix definitivo si 1+2 no bastan: currículo temporal / pesos causales). Secundario:
superficies libres 7.4MPa, σ_VM máx en x=0.2 no en empotramiento. Aplicados (1)+(2)
(config + Dockerfile); pendiente reentrenar; (3) en reserva. Tocados: config, Dockerfile.train; docs 05/BITACORA.

### Fase 7 — 2.º reentreno: colapso trivial → EQUILIBRIO SECCIONAL (2026-06-22) → [05 §4-ter](05_Diagnostico_resultados.md)
Con el régimen de flexión, el **FEM quedó perfecto** (flecha 1429µm oscilando, σ_VM 48MPa
en empotramiento) pero la **PINN colapsó a casi-trivial** (pérdida 2e-5 pero flecha 15µm,
σ_VM 1.3MPa SOLO en la punta, L2 vs FEM 99%). **Causa (modo de fallo intrínseco, no bug):**
flexión controlada por fuerza + EDP homogénea (ρü=∇·σ) → `u≈0` da residuo≈0; la única BC
motora (tracción) es débil → el gradiente se queda en la cuenca trivial y la pérdida total
no lo detecta (mínimo degenerado). **Fix físico-puro:** `pde_loss.loss_equilibrium` impone
el equilibrio INTEGRAL por sección (consecuencia de ∇·σ=0, no datos): `∫τ̂_xy dη=-K_shear·g`
(K_shear=P/Σc=0.02) y `∫σ̂_xx·η dη=+K_moment·(1-ξ)·g` (K_moment=P L/Σc²=0.3), signo/forma
verificados vs FEM. Saca del trivial (con `u≈0`, L_eq=0.022 ×w_eq=20 ≫ resto). Blando para
no rigidizar a cuasi-estático. Implementación: `sampling.sample_sections` (n_sec=400 ×
n_eta_eq=17 nodos η, agrupados por sección), término en `train` sin trocear, fila L_eq en
`app`. Validado py_compile + lote diminuto. **Pendiente reentrenar.** Tocados: config,
sampling, pde_loss, train, app; docs 05/CLAUDE/skill. **Sincronización docs completa**
(régimen de flexión + escalas nuevas + ansatz saturante + residuo ×ĉ + equilibrio
seccional): README + docs 00/01/02/03/04 actualizados y libres de referencias obsoletas
(escala 3P/4c, ansatz (t̂/T̂)², $\hat T$=4/0.77 ms, pesos w_bc=10) salvo en contexto
histórico explícito.

### Fase 6 — Diagnóstico honesto tras reentrenar (2026-06-22) → ver [05_Diagnostico_resultados.md](05_Diagnostico_resultados.md)
Tras reentrenar con restricciones duras, la PINN sigue ~100× blanda (flecha 6.5 µm) y
el error L2 vs FEM es ~90 %. **El problema NO es un bug, son dos escalas mal elegidas:**
1. **Escala temporal**: `T_REF=L√(ρ/E)` es el tiempo de **onda elástica** (tránsito
   0.29 ms), no el de **flexión** (T_bending=13.46 ms, f₁=74 Hz). Con `T_hat=4` el
   horizonte es **0.773 ms = 5.7 % de un periodo de flexión** → la viga no flecta; el
   vídeo del FEM muestra **ondas** (2.7 tránsitos), no flexión. Por eso FEM (68 µm) ≪
   Euler-Bernoulli (803 µm): ni el FEM ha tenido tiempo de flectar.
2. **Escala de desplazamiento**: `U_REF=3.57 µm` (escala de *tensión* local) hace que la
   flecha de flexión adimensional sea **~225** (= (L/c)²) cuando debería ser O(1) → la
   red no puede alcanzarla, menos aún con el *ansatz* `ξ·(t̂/T̂)²` (factor <1).
- **Consecuencias de interpretación**: los checks de Euler-Bernoulli (99.7 %) y de
  frecuencia (1597 %) **no miden la PINN**, son comparaciones inválidas (regímenes
  distintos); la "frecuencia" 1.26 kHz es un **artefacto de ventana FFT**.
- **Matices documentados**: el reescalado del residuo `×ĉ²` no rebalancea términos
  internos (solo baja la magnitud global → enforcement EDP más débil); el *ansatz*
  `(t̂/T̂)²` crece ×16 en la ventana (mal condicionado como envolvente global).
- **Camino**: Opción A (recomendada) = llevar al **régimen de flexión** (T_hat≈75–150,
  reescalar U_REF a flexión, ansatz CI saturante) → PINN≈FEM≈EB triangulan; Opción B =
  asumir **régimen de ondas**, quitar el contraste EB, validar solo vs FEM + causal +
  Fourier features. El estado actual (ondas + validar con flexión) es el peor sitio.
- **Termómetro correcto**: error L2 PINN vs FEM, no la pérdida total (reescalada).
- **DECISIÓN: Opción A (régimen de flexión). IMPLEMENTADO** (validado py_compile +
  forward/loss en lote diminuto; pendiente reentrenar en RunPod): `T_hat=4→100`;
  `SIGMA_SCALE` = tensión de flexión `P L²/(2c³)` (50 MPa) en vez de la tracción
  `3P/4c` → flecha adim. O(1); tracción `tau_hat_peak=(3P/4c)/Σ≈0.015`; ansatz
  saturante `ξ·t̂²/(t̂²+τ²)` con `ic_tau=20` (antes `ξ·(t̂/T̂)²`); `t̂_ramp=1→20`;
  residuo `×ĉ²→×ĉ`; `n_res=4000→6000`, épocas `2000+200→3000+300`. Tabla de cambios y
  checklist post-reentreno en `docs/05_Diagnostico_resultados.md §4-bis`. Tocados:
  config.py, models/pinnsformer.py, physics/pde_loss.py, train.py, utils/inference.py,
  utils/metrics.py; docs 00/05 + CLAUDE.md + skill pinn-debugging.

### Fase 5 — Reestructuración: simplificación + ground truth FEM
Feedback de tutores: "impresionante pero demasiado complejo". Tres directrices:
1. **Escenario fijo** (sin generalización): `config.VIGA` con L, c, P_max, t_ramp fijos;
   solo acero. Red de **d_in=7 → d_in=3**; `Dominio` sin rangos paramétricos;
   `sampling.COLS=["xi","eta","t"]`; `pde_loss`/`inference` leen geometría de `VIGA`;
   `pinnsformer.forward(xi,eta,t)` sin params; `input_bounds` de 3.
2. **Ground truth real por FEM** (`fem/fem_solver.py`, scikit-fem 12 vía `uv add`):
   elastodinámica tensión plana, `ElementVector(P1)` en malla que coincide nodo a
   nodo con la rejilla de inferencia, **Newmark-β** (1/4,1/2), tracción facet x=L
   parabólica, tensiones por proyección L2. `solve_fem()` devuelve el mismo dict que
   `evaluate_fields`. Verificado: σ_VM máx en el **empotramiento** (correcto).
   `metrics.py`: `compare_fields_pinn_fem` (error L2) y `winner_table` (ganador por
   categoría). `app.py` reescrita: radio PINN/FEM, curvas superpuestas, tarjeta 🏆.
3. **Carga suavizada**: sigmoide → **rampa lineal**. No cambia la EDP (solo el factor
   temporal de la BC); idéntica en PINN y FEM.
- Docs actualizadas: README + 00..04. Dudas de validación (flecha estática,
  Euler-Bernoulli, frecuencias) en `docs/03 §12.5`.

### Fase 4 — Diagnóstico del colapso trivial y plan de arreglo (luego aplicado)
- El modelo paramétrico **colapsaba a la solución trivial** (u,v≈0): pérdida ~1.65,
  flecha ~0.3µm vs ~800µm de Euler-Bernoulli, σ_VM máx en la punta (solo la tracción
  impuesta) en vez del empotramiento. Causas: (1) **atractor trivial** (BC+IC
  homogéneas tiran a cero, carga diluida en w_bc); (2) **residuo mal condicionado**
  (U_ηη/ĉ² pesa ~100× más que U_t̂t̂).
- **Correcciones** (sin tocar el PINNsFormer del paper): restricciones duras
  `ξ·(t̂/T̂)²`; peso propio de la carga `w_load=20`; reescalado del residuo ×ĉ².

### Fase 1-3 — Construcción inicial (versión paramétrica)
- PINNsFormer paramétrico (d_in=7: ξ,η,t̂,L̂,ĉ,p̂,t̂_imp) que generalizaba geometría/
  carga sin reentrenar. Adimensionalización completa (clave para converger).
- Pseudo-secuencia temporal del paper (k=5). Entrenamiento Adam→L-BFGS con
  acumulación de gradiente por bloques (la 2.ª derivada agota la VRAM).
- MLOps: RunPod GPU → W&B → app Dash local. Documentación didáctica `docs/00..04`.
- Métricas y validación multinivel (`utils/metrics.py`): sim metrics, sanity checks,
  Euler-Bernoulli, residuo en malla nueva.

> La versión paramétrica se documenta como compromiso de diseño (simplicidad ↔
> generalización) y es recuperable ampliando `VIGA`/`Dominio` y `d_in`.
