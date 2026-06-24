# 🔬 05 — Diagnóstico honesto de los resultados (2026-06-22)

> Documento de análisis tras el reentrenamiento con restricciones duras. Responde a
> tres preguntas: **(1) ¿qué falla?**, **(2) ¿cómo sería un resultado bueno y por
> qué?**, **(3) ¿qué camino seguir?**. Las cifras salen de `config.py` + las fórmulas
> del código, no de fuentes externas (verificadas con cálculo cerrado).

## 0. TL;DR

El modelo **no está "roto" por un bug**: está atrapado entre **dos problemas de
escala independientes** que vienen del *planteamiento*, no del entrenamiento.

1. **Escala temporal mal elegida.** El horizonte simulado es **0.773 ms**, que es el
   régimen de **ondas elásticas**, no el de **flexión** de la viga. La viga necesita
   **~13.5 ms** (un periodo de flexión) para flectar de verdad. Simulamos **el 5.7 %
   de un periodo**: la viga apenas ha empezado a moverse.
2. **Escala de desplazamiento mal elegida.** La adimensionalización fija `U_REF`
   según la *tensión local*, pero la **flecha de flexión es 225× mayor** que `U_REF`.
   La red tendría que producir salidas adimensionales de **orden ~225** (deberían ser
   orden 1) → imposible de alcanzar a través del *ansatz* de restricción dura.

Consecuencia: **las métricas de contraste (Euler-Bernoulli, "frecuencia") no son
fallos de la PINN, son comparaciones inválidas en este régimen**; y el desacuerdo
PINN↔FEM (90 %) sí es real pero se explica por (2) + la dificultad conocida de las
PINN con ondas.

---

## 1. La pista que lo destapa todo: FEM y Euler-Bernoulli discrepan ×12

| Magnitud | PINN | FEM (ground truth) | Euler-Bernoulli (analítico) |
|---|---|---|---|
| Flecha máx. extremo | 6.5 µm | 68.4 µm | **803.6 µm** |
| "Frecuencia" | 1.26 kHz | — | **0.074 kHz (74 Hz)** |

Si el FEM fuese una verdad de referencia *del problema de flexión*, debería coincidir
con Euler-Bernoulli (ambos resuelven la misma viga). Discrepan ×12. **No es que el FEM
esté mal**: es que **ni el FEM ha tenido tiempo de flectar**. La clave está en los
tiempos.

### Las dos escalas temporales del problema (verificadas por cálculo)

```
T_ref  = L·√(ρ/E)               = 0.1933 ms   ← tiempo de TRÁNSITO DE ONDA elástica
T_phys = T_hat · T_ref (T_hat=4) = 0.7734 ms  ← horizonte total simulado
c_L    = √(E/ρ) = 5172 m/s  →  tránsito de onda a lo largo de L = 0.290 ms
   → en 0.773 ms la onda cruza la viga 2.7 veces

T_bending (1er modo de flexión, Euler-Bernoulli) = 13.46 ms   (f₁ = 74.3 Hz)
   → T_phys / T_bending = 0.057   (¡simulamos el 5.7 % de UNA oscilación!)
```

**`T_REF = L·√(ρ/E)` es el tiempo de la onda de presión, NO el de la flexión.** La
flexión está gobernada por la rigidez `EI` y es ~50× más lenta. Al fijar el horizonte
en `T_hat=4` (≈ 0.77 ms) el problema vive **en el régimen de ondas**, donde:

- La viga **no flecta** (no le da tiempo) → flecha FEM 68 µm ≪ 800 µm estáticos.
- Lo que se ve en el vídeo del FEM son **ondas de tensión propagándose y rebotando**
  (2.7 tránsitos), no flexión. Por eso la viga "no se troncha".
- La **"frecuencia" medida por FFT (1.26 kHz, periodo 0.79 ms) es un artefacto**: con
  una ventana de 0.773 ms es físicamente imposible resolver una oscilación de 13.5 ms;
  la FFT devuelve básicamente *el inverso de la ventana* (~0.79 ms → 1.26 kHz). No es
  ninguna frecuencia natural.

### Por qué los "errores" del Nivel 3 son engañosos

| Check (Nivel 3) | Valor | Veredicto real |
|---|---|---|
| Error flecha vs Euler-Bernoulli | 99.7 % | **Comparación inválida**: EB es teoría cuasi-estática de flexión; el caso está en régimen de ondas. Nunca puede pasar. |
| Error frecuencia vs EB | 1597 % | **Comparación inválida**: la "frecuencia PINN" es un artefacto de ventana. |

> ⚠️ Estos dos checks **no miden la calidad de la PINN** en el caso actual; miden la
> distancia entre dos regímenes físicos distintos. Han estado induciendo a pensar que
> la PINN falla mucho más de lo que realmente falla en *su propio* problema.

---

## 2. El segundo problema (independiente): la flecha pide salidas de orden ~225

```
U_REF = L·σ_SCALE/E = 3.57 µm          ← escala de desplazamiento del config
Flecha de flexión / U_REF = 803.6/3.57 = 225   ← adimensional que exige la flexión
ratio geométrico (L/c)² = (1.5/0.1)² = 225      ← coincide: es amplificación de flexión
```

`U_REF` se eligió a partir de la **tensión axial local** (`σ_SCALE = 3P/4c`), que es la
escala correcta para **tensiones** y para el **régimen de ondas** (donde los
desplazamientos son pequeños). Pero la **flecha de flexión** se amplifica por el factor
geométrico `(L/c)²=225`: en variables adimensionales la flecha vale **~225**, cuando una
PINN bien escalada debería producir salidas **de orden 1**.

Encima, el *ansatz* de restricción dura multiplica la salida por `ξ·(t̂/T̂)²`, un factor
**< 1** en casi todo el dominio. Para que `V·factor ≈ 225` la red cruda tendría que
emitir valores aún mayores → el optimizador no llega y **se queda corto** (flecha PINN
6.5 µm ≈ U_REF·O(1), es decir, la red produce orden 1 como "debería" estar escalada,
pero la física pide 225). **Esto explica por qué la PINN sale ~100× blanda.**

> Nota técnica sobre el *ansatz* `(t̂/T̂)²`: crece cuadráticamente en toda la ventana
> (×16 entre t̂=0 y t̂=4). Solo es razonable para imponer la CI cerca de t̂=0; como
> envolvente global mal-condiciona el problema. Conviene un factor que valga ~t̂² cerca
> de 0 pero **sature a 1** después (p. ej. `1-(1+t̂/τ)e^{-t̂/τ}`).

> Nota sobre el reescalado del residuo `×ĉ²`: multiplica **todo** `r_u` por la misma
> constante, así que **no rebalancea los términos internos** (U_ηη/ĉ² sigue pesando
> ~100× más que U_t̂t̂ *dentro* de la ecuación); solo **baja la magnitud global** del
> residuo frente a las pérdidas de contorno → enforcement de la EDP más débil. El
> rebalanceo real, si se quiere, debe ser término a término o vía pesos adaptativos.

---

## 3. Cómo sería un resultado BUENO y POR QUÉ

Hay que **elegir régimen**. Las dos opciones dan proyectos defendibles, pero distintos.

### Opción A (recomendada) — Régimen de FLEXIÓN
Llevar el problema a donde **PINN, FEM y Euler-Bernoulli coinciden** → validación
triangulada, limpia y muy defendible. Requiere **3 cambios de escala** (no tocan la
arquitectura del paper):

1. **Horizonte temporal** que cubra ≥1 periodo de flexión: `T_hat ≈ 75–150`
   (T_phys ≈ 14–28 ms). Solución suave y de baja frecuencia → *régimen donde las PINN
   funcionan bien*.
2. **Reescalar `U_REF`** a la escala de flexión (`U_REF_flex = P·L³/(EI)` ó equivalente)
   para que la flecha adimensional sea **O(1)**.
3. **Ansatz de CI saturante** (no `t̂²` creciente) y carga en **rampa lenta** (que la
   rampa dure una fracción del periodo de flexión, no del de onda).

**Qué se vería si está bien:**

| Señal | Valor objetivo (y por qué) |
|---|---|
| Flecha PINN ≈ FEM ≈ EB | ~**800 µm** estática (dinámica hasta ~1.5–2× si el escalón es brusco). Las tres a < 20 %. |
| Frecuencia ≈ 74 Hz en las tres | el horizonte ya resuelve ≥1 periodo → la FFT mide la **flexión real**, no la ventana. |
| σ_VM máx en el **EMPOTRAMIENTO** | ~**22.5 MPa** (= M·c/I con M=P·L). Momento flector máximo en x=0. |
| Error L2 PINN vs FEM | **< 20–30 %** en desplazamiento y σ. |
| Vídeo | la viga **flecta y oscila** (la punta sube/baja); banda **roja de tensión en el empotramiento** (fibras superior/inferior) que **oscila** en el tiempo. *No* ondas: flexión suave. |

### Opción B — Régimen de ONDAS (el actual, reformulado honestamente)
Aceptar que es **propagación de ondas elásticas** y reescribir el TFM como tal:

1. **Eliminar el contraste Euler-Bernoulli** (no aplica) → validar **solo contra FEM**.
2. Invertir en técnicas de PINN para ondas: **entrenamiento causal** (pesar/curriculum
   en el tiempo: primero t pequeños), **features de Fourier** en la entrada para vencer
   el *spectral bias*, más resolución temporal, más épocas.

**Qué se vería si está bien:** el vídeo de la PINN **reproduce los frentes de onda del
FEM** (viajan a c_L≈5172 m/s, rebotan); desplazamientos pequeños (decenas de µm);
tensiones dominadas por los frentes. Error L2 PINN vs FEM < 30 %.

> ⚠️ La Opción B es **honesta pero dura**: las PINN *vanilla* (incluida PINNsFormer)
> son notoriamente malas con problemas hiperbólicos/ondas (frentes = alta frecuencia =
> *spectral bias*; falta de causalidad). Llegar a < 30 % L2 cuesta esfuerzo real.

### Por qué el estado actual es el PEOR sitio posible
Estamos en **régimen de ondas (difícil para PINN) PERO validando contra teoría de
flexión (no aplica)**: lo peor de los dos mundos. Cualquiera de las dos opciones es
estrictamente mejor.

---

## 4. Cómo MEDIR si vamos bien (cambiar el termómetro)

- **La pérdida total NO es el termómetro.** 2.0e-3 "parece" bajo pero el residuo está
  reescalado ×ĉ²≈0.01 (lo encoge artificialmente) y un campo casi trivial puede dar
  pérdida baja. **El termómetro es el error L2 PINN vs FEM** (sección "Concordancia de
  campos"): ahí están los 90 %, y *ese* es el número a bajar.
- **Triangulación** (solo en régimen de flexión): PINN ≈ FEM ≈ EB en flecha, frecuencia
  y σ_VM-en-empotramiento. Si las tres coinciden, el proyecto está validado de tres
  formas independientes (analítica, numérica clásica, y PINN).
- **Ubicación de σ_VM máx**: debe migrar de la *punta* (solo la tracción impuesta) al
  *empotramiento* (momento flector). Hoy está mal por estar en régimen de ondas.
- **Residuo en malla (Nivel 4)** sobre puntos no vistos, *sin* el reescalado ×ĉ², como
  control de que la EDP se cumple de verdad en el interior.

---

## 4-bis. CAMBIOS YA IMPLEMENTADOS (Opción A · régimen de flexión) — 2026-06-22

Validado con `py_compile` + forward/loss en lote diminuto (CPU). **Pendiente de
reentrenar en RunPod.** Resumen de qué se tocó y con qué valor:

| Cambio | Antes | Ahora | Fichero | Por qué |
|---|---|---|---|---|
| Horizonte temporal | `T_hat=4` (0.77 ms) | `T_hat=100` (≈1.44 periodos, 19 ms) | `config.DOMINIO` | meter el caso en el régimen de flexión |
| Escala de tensión | `3P/4c` (0.75 MPa) | `P L²/(2c³)` (50 MPa) = flexión | `config.SIGMA_SCALE` | flecha adim. O(1) (antes ~225); σ adim. O(1) |
| Tracción aplicada | `p̂·ĉ_REF/ĉ` (=1) | `τ̂_peak=(3P/4c)/Σ` (≈0.015) | `Viga.tau_hat_peak`, `pde_loss.applied_traction` | coherente con la nueva Σ |
| Ansatz CI | `ξ·(t̂/T̂)²` (crece ×1e4) | `ξ·t̂²/(t̂²+τ²)` (satura a 1), `τ=20` | `pinnsformer.forward`, `ModeloCfg.ic_tau` | no descondicionar sobre horizonte largo |
| Rampa de carga | `t̂_ramp=1` (= onda) | `t̂_ramp=20` (≈0.3 periodos) | `config.VIGA` | excitar flexión, no ondas |
| Reescalado residuo | `×ĉ²` (hunde a ~1e-4) | `×ĉ` (mean(r²)~load/bc) | `pde_loss.loss_residual` | no infra-imponer la EDP |
| Colocación / épocas | 4000 / 2000+200 | 6000 / 3000+300 | `config.ENTRENAMIENTO` | ventana temporal 25× mayor |

`inference.py` no necesitó cambios de escala: reescala desde `SIGMA_SCALE` y
`material.U_ref` (que se actualizan solos). El **FEM va en SI puro** → no le afecta el
cambio de adimensionalización; sigue siendo el *ground truth*.

**Qué revisar tras el reentreno (en orden):** (1) ¿la flecha PINN se acerca a FEM y a
EB (~800 µm)? (2) ¿σ_VM máx migra al **empotramiento** (~22 MPa)? (3) ¿el **error L2 vs
FEM** baja de 90 % a < 30 %? (4) si la flecha sigue corta, el **balance de pérdidas** es
el primer knob: ver que `res/load/bc` salgan del mismo orden y subir `w_res`.

## 4-ter. SEGUNDO reentreno (régimen de flexión) → COLAPSO TRIVIAL → equilibrio seccional

Tras el cambio de régimen, el **FEM quedó perfecto** (flecha 1429 µm oscilando, σ_VM
48 MPa en el empotramiento), pero la **PINN colapsó a una solución casi trivial**:
pérdida bajísima (2e-5) pero flecha 15 µm, σ_VM 1.3 MPa **sólo en la punta** (= la
tracción impuesta), error L2 vs FEM 99 %. La red reproduce la tracción de la punta en
una capa fina y deja `u≈0` en el interior.

**Causa (modo de fallo intrínseco, NO un bug de escala):** el problema es de flexión
**controlada por fuerza** y la EDP del interior es **homogénea** (ρü=∇·σ, sin fuerza de
volumen) → `u≈0` la satisface con residuo ≈0. Lo único que debería generar la gran
flexión es una BC de Neumann **débil** (τ̂≈0.015 en la punta). El descenso por gradiente
se queda en la cuenca trivial. La pérdida total NO lo detecta (mínimo degenerado): por
eso el semáforo decía "BUENO" con un resultado pésimo. Es un fallo conocido de las PINN
*vanilla* en elasticidad accionada por fuerza (la FEM no lo sufre porque su matriz de
rigidez global acopla todo; la PINN sólo tiene residuo local).

**Fix — EQUILIBRIO SECCIONAL (física pura, `pde_loss.loss_equilibrium`):** se impone el
equilibrio INTEGRAL que el residuo local no transmite. En toda sección ξ:

```
 V̂(ξ,t̂) = ∫₋₁¹ τ̂_xy dη     = -K_shear · g(t̂)        K_shear  = P/(Σ·c)   = 0.02
 M̂(ξ,t̂) = ∫₋₁¹ σ̂_xx·η dη   = +K_moment·(1-ξ)·g(t̂)   K_moment = P·L/(Σ·c²) = 0.30
```

Son **consecuencias exactas de ∇·σ=0** (no son datos), y fijan directamente la amplitud
de flexión: con `u≈0` el momento integral da 0 en vez de 0.3(1-ξ) → penalización grande
→ la red sale del trivial. Signo y forma **verificados contra el FEM** (M baja lineal de
0.26 en el empotramiento a 0 en la punta; V≈-0.017). Se impone **blando** (`w_eq=20`):
fija el esqueleto cuasi-estático; la desviación dinámica (inercia, sobreoscilación) la
aporta el residuo. `g(t̂)=min(t̂/t_ramp,1)` = nivel cuasi-estático de la carga.

Implementación: `utils/sampling.sample_sections` (n_sec=400 pares ξ,t̂ × n_eta_eq=17
nodos η, agrupados por sección), `pde_loss.loss_equilibrium` (cuadratura trapezoidal en
η), término `w_eq` en `train` (sin trocear: la integral exige secciones completas), fila
`L_eq` en `app`. **Verificado**: con `u≈0`, `L_eq=0.022` (×20 = 0.44 ≫ resto) → el
colapso deja de ser mínimo. **Pendiente de reentrenar.**

**Qué revisar tras este reentreno:** (1) `L_eq` debe BAJAR mucho (la red ya genera el
momento) — si se queda alto, subir `w_eq`; (2) flecha → centenares de µm; (3) σ_VM máx
al **empotramiento** ~40-50 MPa; (4) **L2 vs FEM < 30 %**; (5) ¿aparece la oscilación de
la punta? (la da el residuo sobre el esqueleto del equilibrio). Si la respuesta sale
demasiado cuasi-estática (sin oscilar), BAJAR `w_eq`.

## 4-quater. TERCER reentreno → la PINN flexa pero NO oscila (solución cuasi-estática)

El equilibrio seccional **rompió el colapso**: flecha 843 µm (≈ EB estática 803, error
12.9 %), σ_VM 27 MPa cerca del empotramiento, L2 vs FEM 99 %→**51 %**. Pero la viga
**no rebota**: baja a la flecha estática y se queda; la "frecuencia 0.05 kHz" es de nuevo
el artefacto de ventana (periodo 19.8 ms ≈ ventana) → **no hay oscilación real**.

**¿Por qué el FEM sí rebota?** Es **elastodinámica**: la viga es un oscilador (masa +
rigidez). Al aplicar la carga, la **inercia** la hace pasarse de largo de su equilibrio
estático (sobreoscilación dinámica, hasta ~2× la flecha estática) y luego volver, vibrando
en torno a los ~800 µm a su 1.ª frecuencia natural (~74 Hz, periodo 13.5 ms). Sin
amortiguamiento, sigue oscilando. **Es el resultado esperado y correcto**: la flecha
estática (~800 µm) es la *media*; el rebote es la parte *dinámica*.

**Diagnóstico — la PINN convergió a la solución CUASI-ESTÁTICA, no a la dinámica.** Acertó
el esqueleto de flexión (EB al 13 %) pero le falta la oscilación. El 51 % de L2 es casi
todo eso: el FEM barre 0↔1428 µm mientras la PINN se queda en ~800. **NO es un fallo
estructural del método ni "el FEM es mejor en esencia"** (la PINN ya domina la física de
flexión); es la **dinámica transitoria**, que falla por tres causas, en orden de impacto:

1. **`w_eq=20` era demasiado alto y su objetivo es CUASI-ESTÁTICO.** El momento objetivo
   `M̂=K·(1-ξ)·g(t̂)` es, tras la rampa (g=1), **constante en el tiempo** → prohíbe que el
   momento oscile → fuerza la solución cuasi-estática. Salvó del colapso pero ahora
   sobre-restringe. **Fix: `w_eq` 20→5** (sigue rompiendo el trivial, deja oscilar al
   residuo).
2. **Infra-entrenamiento: 2000 Adam / 200 L-BFGS.** El `Dockerfile.train` fijaba
   `ENV ADAM=2000` por variable de entorno, que **anulaba** el config (3000) — el reentreno
   usó 2000/200. La oscilación es contenido temporal que tarda en ajustarse (sesgo
   espectral). **Fix: Dockerfile `ADAM=6000, LBFGS=400, MAX_MINUTES=90`.**
3. **Sin entrenamiento causal (la razón de fondo, semi-estructural).** Una PINN *vanilla*
   entrenada con todos los instantes a la vez tiende a converger a la solución
   **promediada en el tiempo** = justo la cuasi-estática que vemos (Wang et al.,
   *"Respecting causality"*). El residuo es además poco sensible a la oscilación (la
   cuasi-estática casi lo satisface). **Fix definitivo si 1+2 no bastan: entrenamiento
   CAUSAL / currículo temporal** (ventana de tiempo creciente, o pesar primero los t
   pequeños), para que la red construya el transitorio en orden causal.

**Secundario:** superficies libres violadas (7.4 MPa) y σ_VM máx. ligeramente adentro
(x=0.2) en vez del empotramiento → campo de tensión no perfectamente limpio (la
sobreoscilación que concentra 48 MPa en el empotramiento no se captura). Más épocas y, si
hace falta, subir `w_bc` lo afinan.

**Plan:** reentrenar con (1)+(2) —cambios baratos, ya aplicados—; si la punta aún no
rebota, implementar (3) entrenamiento causal. **No es que el FEM gane al método**: para
*este* caso dinámico 2D el FEM es más preciso y ~100× más rápido (esperado), pero el 51 %
es dinámica recuperable, no un muro fundamental.

## 4-quinquies. CUARTO reentreno → CAUSALIDAD: por qué no oscila y cómo forzarlo

El 3.er reentreno (con `w_eq=5`) confirmó: la PINN flexa bien (833 µm, EB 12.7 %) pero
**no oscila** (la punta baja y se queda); L2 vs FEM se estanca en ~52 %. Más épocas o
pesos adaptativos NO lo arreglan. Razón de fondo (clave):

> La oscilación es la **vibración libre** de la viga (modo propio excitado al reconciliar
> la deflexión de la carga con el reposo inicial). **Un modo libre tiene residuo de EDP
> = 0** → la pérdida de residuo es INDIFERENTE a si la red lo incluye o no. La solución
> **cuasi-estática** (baja y se queda) satisface residuo≈0, BCs, IC (el ansatz `t̂²`
> fingía u̇(0)=0) **y** `L_eq` (cuyo objetivo *es* cuasi-estático). Es un mínimo legítimo
> de las 5 pérdidas, y el "más simple" → el optimizador aterriza ahí. La pérdida **no
> puede distinguir** la cuasi-estática de la oscilante. El FEM sí oscila porque **marcha
> en el tiempo** (Newmark): la causalidad está incorporada.

No es ni infra-entrenamiento ni desbalance de pesos. La cura es **inyectar causalidad
temporal**. Implementadas DOS estrategias complementarias (sin tocar las capas del paper):

**(3) Quitar el sesgo cuasi-estático** — para que la oscilación deje de ser opcional:
- **CI de velocidad BLANDA** (`models/pinnsformer.py`, `MODELO.hard_ic=False`): el ansatz
  pasa de `ξ·t̂²/(t̂²+τ²)` (anula desplazamiento Y velocidad) a `ξ·t̂/(t̂+τ)` (anula sólo el
  **desplazamiento**; u̇(0)≠0 por construcción). Así la red debe imponer `u̇(0)=0` como
  pérdida (`w_ic=10`, antes 1) → reconciliar globalmente carga ↔ reposo es justo lo que
  **excita el modo libre**. (Verificado: u(0)=0 exacto, u̇(0)≠0.)
- **`L_eq` menor y con relajación temporal** (`physics/pde_loss.py`, `w_eq` 5→3): el
  constraint actúa pleno durante la rampa (t̂≤t_ramp, fija el esqueleto) y se DESVANECE
  después (`exp(-(t̂-t_ramp)/eq_relax_tau)`) → ya no impone el momento estático constante
  en t̂ tardío, dejando libre la oscilación. Sigue rompiendo el colapso trivial.

**(2) Currículo temporal** (`config.curriculum_stages`, `train.py`): el Adam se entrena en
ETAPAS con ventana `[0, t_frac·T_hat]` creciente `((0.3,0.3),(0.6,0.3),(1.0,0.4))` —
(fracción de tiempo, fracción de épocas). La red clava primero el transitorio inicial y lo
extiende, **respetando la causalidad** como el FEM. `sampling.build_training_sets` acepta
`t_max`; el "mejor" modelo se decide sólo en la ventana completa (última etapa).

**Pesos adaptativos (NTK/SA-PINN): NO implementados** — atacan el desbalance entre
términos, no esta degeneración cuasi-estática↔oscilante; útiles sólo para pulir. **Pendiente
reentrenar** y comprobar si la punta ya rebota (curva `v(L,0,t)` oscilando como el FEM) y si
L2 baja de 50 %. Si aún no basta, el siguiente paso serían **pesos causales** (Wang et al.,
ponderar el residuo por `exp(-ε·Σ_{t'<t} residuo)`), más fino que el currículo.

### §4-sexies — La relajación de L_eq fue un error; diagnóstico real del no-rebote

El 5.º reentreno (causalidad de §4-quinquies, `w_eq=3` + relajación temporal de `L_eq`)
**empeoró** la estática en lugar de añadir rebote. Síntomas medidos: flecha estática 595µm
(antes 833 con `w_eq=5`; EB 803), **σ_VM máx. en el centro** en vez del empotramiento y a
~40 % de magnitud (19 vs 48 MPa), sin oscilación. Análisis:

1. **La relajación apagaba el esqueleto de tensión.** `relax=exp(-(t̂-t_ramp)/τ)` con τ=20
   reduce `L_eq` a 0.14 en t̂=60 y a 0.018 en t̂=100 → el constraint está prácticamente
   APAGADO en el 80 % de la ventana. Pero `L_eq` es el único término que impone la FORMA
   del momento `M̂∝(1-ξ)` (máx. en empotramiento): la EDP local homogénea no transmite la
   distribución global. Sin él, la tensión se degrada y su máximo migra al centro. La forma
   del momento es física EXACTA a TODO instante (∇·σ=0), no sólo durante la rampa →
   **revertida**: `L_eq` pleno en todo t̂, `w_eq` 3→5.

2. **Por qué no rebota (la causa de fondo, de condicionamiento).** La oscilación de flexión
   es un modo MUY blando: en unidades de tiempo de ONDA (`T_REF`), su frecuencia adimensional
   es `ω̂=2π/periodo_t̂ ≈ 0.09`, de modo que la señal inercial que separa "oscila" de
   "cuasi-estático" es `~ω̂²≈8e-3`. Pero el suelo de residuo que alcanza el optimizador es
   `L_res=3e-4 → RMS≈1.7e-2`, **mayor que esa señal**. Minimizando `mean(r²)` global en el
   tiempo, el rebote queda *enterrado bajo el propio ruido de convergencia*: por eso ningún
   ajuste de pesos/épocas lo ha movido. Es mis-escalado, no irresoluble.

3. **L-BFGS truncado.** El reentreno cortó L-BFGS a 40/100 épocas (límite de tiempo). L-BFGS
   es la fase de PRECISIÓN que baja `L_res` 1-2 órdenes — justo lo necesario para que el suelo
   caiga por debajo de `ω̂²` y el rebote emerja. Se entrenó sin la fase decisiva.

**Plan (certeza decreciente):** (1) revertir la relajación + `w_eq=5` *[hecho]* recupera la
estática y la tensión en el empotramiento; (2) **garantizar que L-BFGS termine** las 100
épocas (reasignar presupuesto: menos Adam, mismo dinero); (3) reserva si persiste: re-escalar
`t̂` por el periodo de FLEXIÓN (no el de onda) → la inercia pasa a O(1) y el rebote deja de
estar bajo el suelo de residuo. **Pesos causales: NO** — reordenan *cuándo* se entrena cada
instante, no cambian que la señal esté bajo el suelo; no atacan esta degeneración.

### §4-septies — El muro NO es el escalado temporal; es la causalidad (pesos causales)

Tras el fix de §4-sexies la **estática quedó correcta** (flecha 703µm, EB 803 → **12.5 %** ✓,
σ_VM volviendo al empotramiento) pero **seguía sin rebotar** (L2 ~52-54 %, "periodo"=ventana,
artefacto de FFT sobre una caída monótona). Auditoría de la causa y descarte de hipótesis:

1. **Re-escalar t̂ por el periodo de flexión NO sirve (demostrado).** El residuo físico
   `R=∂σ/∂x−ρ·∂²u/∂t²` es invariante de unidades: cambiar la escala de tiempo sólo lo
   multiplica por una constante y reetiqueta el eje. La señal que separa oscilar/no-oscilar
   es `ω̂²≈c²/L²·cte≈0.018`, un **adimensional físico** (esbeltez²); ninguna reescala lo
   cambia. La inercia pasa de "`U_t̂t̂` pequeño, coef. 1" a "`U_t̃t̃`∼1, coef. 0.018": el mismo
   0.018. **Habría sido gasto inútil de cómputo.**

2. **El muro real (modo de fallo documentado).** Krishnapriyan et al. (NeurIPS 2021) y Wang
   et al. (2022): las PINN globales-en-tiempo fallan en transitorios oscilatorios porque la
   solución cuasi-estática y la dinámica tienen residuo casi idéntico (difieren en `~ω̂²`, bajo
   el suelo `RMS≈0.017`). La pérdida global no las distingue y elige la cuasi-estática, que
   además **viola la CI de reposo** (`u̇_qs(0)=u_est/t_ramp≈0.05≠0`). El FEM acierta porque
   **marcha en el tiempo** (Newmark integra la inercia paso a paso).

3. **Fix: pesos causales** (`config.causal`, `pde_loss._causal_residual`). Se pondera el
   residuo por franjas temporales con `w_i=exp(-ε·Σ_{j<i}L_j)` (peso detenido del grafo): cada
   franja t̂ sólo cuenta cuando las anteriores están resueltas → la red satisface el residuo en
   ORDEN temporal, propagando el reposo inicial hacia adelante (emula el marchado del FEM) y
   haciendo emerger el rebote. Una sola red, una sola corrida (barato). Complementa al currículo
   (que crece la ventana) ordenando dentro de ella. Validado el binning en CPU (autopaceado:
   foco en t=0 si el residuo temprano es alto; activa franjas posteriores al resolverse).

> Nota de honestidad metodológica: en §4-quinquies se desestimaron los pesos causales por un
> razonamiento incompleto ("reordenan cuándo, no la amplitud"). Es incorrecto: la solución
> dinámica **es** la causal (única consistente con arrancar de reposo), luego el pesado causal
> es precisamente lo que la selecciona. Corregido aquí.

**Reserva (si los pesos causales no bastan):** seq2seq con traspaso de estado (red por ventana,
CI = estado final de la anterior; Krishnapriyan 2021) — más pesado, FEM-like, pero el más robusto.

## 5. Resumen para el tribunal (honestidad como fortaleza)

Saber *por qué* un resultado es malo y *cómo medirlo* es nivel de máster. El relato:
"detecté que el caso estaba planteado en el régimen de ondas mientras se validaba
contra teoría de flexión; lo demostré con las escalas temporales (0.77 ms de onda vs
13.5 ms de flexión) y de desplazamiento (flecha adimensional ~225 vs O(1) esperado);
corregí las escalas y entonces PINN, FEM y Euler-Bernoulli triangulan". Eso es un TFM
sólido, no un fallo.
