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

## 5. Resumen para el tribunal (honestidad como fortaleza)

Saber *por qué* un resultado es malo y *cómo medirlo* es nivel de máster. El relato:
"detecté que el caso estaba planteado en el régimen de ondas mientras se validaba
contra teoría de flexión; lo demostré con las escalas temporales (0.77 ms de onda vs
13.5 ms de flexión) y de desplazamiento (flecha adimensional ~225 vs O(1) esperado);
corregí las escalas y entonces PINN, FEM y Euler-Bernoulli triangulan". Eso es un TFM
sólido, no un fallo.
