# 🔧 Elastodinámica de Viga 2D con PINNsFormer

Resolución y visualización de la **elastodinámica 2D de una viga empotrada**
bajo carga de impacto, mediante la arquitectura **PINNsFormer**
(Zhao et al., *ICLR 2024*, [arXiv:2307.11833](https://arxiv.org/abs/2307.11833)),
implementada fielmente a partir del repositorio oficial (`TFM/pinnsformer`).

El proyecto está en su versión **simplificada**: resuelve **un único escenario
fijo** (geometría, carga y material constantes, ver `config.VIGA`) y lo **valida
contra una solución de referencia por Elementos Finitos** (`scikit-fem`),
comparando ambos métodos gráfica y métricamente. Incluye un **dashboard
interactivo en Dash** con animación de la malla deformada (conmutable PINN/FEM),
gráficas superpuestas y una tabla de "ganador por categoría".

> **Nota de evolución.** Una versión anterior implementaba una red *paramétrica*
> ($d_{in}=7$) que generalizaba a cualquier geometría/carga sin reentrenar.
> Siguiendo el feedback de los tutores (*"impresionante, pero demasiado
> complejo"*) se ha **simplificado a un caso fijo** ($d_{in}=3$) y se ha añadido
> el **ground truth por FEM**. La versión paramétrica se puede recuperar
> ampliando `config.VIGA`/`Dominio` y `d_in`.

---

## 1. Física del problema

Viga continua 2D, dominio espacial $x \in [0, L]$, $y \in [-c, c]$ (peralto $2c$),
temporal $t \in [0, T]$. Material elástico lineal (Acero por defecto).

**Ecuaciones de Navier-Cauchy dinámicas (tensión plana):**

$$\mu\,(u_{xx}+u_{yy}) + (\lambda+\mu)\,\frac{\partial}{\partial x}(u_x+v_y) = \rho\,u_{tt}$$

$$\mu\,(v_{xx}+v_{yy}) + (\lambda+\mu)\,\frac{\partial}{\partial y}(u_x+v_y) = \rho\,v_{tt}$$

con $\mu = E\,/\,(2(1+\nu))$, $\lambda = E\nu\,/\,(1-\nu^2)$ (tensión plana) y las
tensiones $\sigma_{xx}$, $\sigma_{yy}$, $\tau_{xy}$ para las condiciones de contorno.

**Condiciones de contorno e iniciales:**

| Frontera | Condición |
|---|---|
| Inicial ($t=0$) | reposo absoluto: $u=v=0$, $u_t=v_t=0$ |
| Empotramiento ($x=0$) | $u=v=0$ |
| Superficies libres ($y=\pm c$) | $\sigma_{yy}=0$, $\tau_{xy}=0$ |
| Extremo cargado ($x=L$) | $\sigma_{xx}=0$, $\tau_{xy} = -\tfrac{3P(t)}{4c^3}(c^2-y^2)$ |

Carga en **rampa lineal**: $P(t) = P_{\max}\cdot g(\hat t)$ con
$g(\hat t) = \min(\hat t / \hat t_{\mathrm{ramp}},\,1)$ — sube de 0 a $P_{\max}$ en
$\hat t \in [0,\hat t_{\mathrm{ramp}}]$ y se mantiene. Perfil parabólico en $y$
(evita singularidades). Arranca exactamente en 0, coherente con el reposo inicial.
El perfil temporal es un **factor de la condición de contorno**: cambiarlo (antes
una sigmoide) **no altera la EDP**, sólo el forzamiento. El **mismo** $g(\hat t)$
se usa en la PINN y en el FEM, para que la comparación sea justa.

### Adimensionalización (clave para la convergencia)

Trabajar en unidades SI es **inviable** para una PINN (acero: $E \approx 2\cdot10^{11}$ Pa,
desplazamientos $\sim10^{-6}$ m). Todo el problema se resuelve en variables
adimensionales (ver `config.py`):

$$\xi = \frac{x}{L} \in [0,1], \qquad \eta = \frac{y}{c} \in [-1,1], \qquad \hat{t} = \frac{t}{T_{\mathrm{ref}}}$$

$$U = \frac{u}{U_{\mathrm{ref}}}, \qquad V = \frac{v}{U_{\mathrm{ref}}}$$

$$T_{\mathrm{ref}} = L_{\mathrm{ref}}\sqrt{\frac{\rho}{E}}, \qquad \sigma_{\mathrm{scale}} = \frac{P_{\mathrm{ref}}\,L_{\mathrm{ref}}^2}{2\,c_{\mathrm{ref}}^3}, \qquad U_{\mathrm{ref}} = \frac{L_{\mathrm{ref}}\,\sigma_{\mathrm{scale}}}{E}$$

Así los coeficientes de la EDP quedan de orden unidad
($\hat{\mu} = 1/(2(1+\nu)) \approx 0.385$, $\hat{\lambda} = \nu/(1-\nu^2) \approx 0.33$).
La inferencia **reescala** la solución adimensional a unidades físicas mediante
$U_{\mathrm{ref}}$ y $T_{\mathrm{ref}}$ (material único: **acero**).

> **Escala de tensión = FLEXIÓN, no la tracción aplicada** (clave, ver
> [docs/05](docs/05_Diagnostico_resultados.md)). Una tracción cortante pequeña en la
> punta genera, por el brazo de palanca $(L/c)$, una tensión de flexión $\sim30\times$
> mayor en el empotramiento y una flecha $\sim(L/c)^2\!\approx\!225\times$ esa escala. Si
> se adimensionaliza por la tracción ($3P/4c$), el desplazamiento adimensional vale
> $\sim225$ (inalcanzable para la red). Por eso $\sigma_{\mathrm{scale}}=P L^2/(2c^3)$
> (la tensión de flexión $M c/I$): así desplazamiento y tensión quedan **de orden 1**.

### Régimen temporal: ondas vs. flexión

$T_{\mathrm{ref}}=L\sqrt{\rho/E}$ es el tiempo de **onda elástica** (la onda cruza la
viga en $\sim0.29$ ms), **no** el de la **flexión** (periodo $\sim13.5$ ms, $\sim50\times$
más lento). El horizonte debe cubrir $\geq1$ periodo de flexión: por eso
$\hat T = 100$ ($\approx1.44$ periodos, $\approx19$ ms). Con el valor anterior ($\hat T=4$,
0.77 ms) la viga ni flexaba —sólo se veían ondas— y la validación contra Euler-Bernoulli
(teoría de flexión) era inválida. El FEM usa la **misma** ventana: comparación justa.

---

## 2. Arquitectura PINNsFormer

Implementación fiel del repo oficial (`models/pinnsformer.py`):
**WaveAct** (activación de onda $w_1\sin+w_2\cos$) → `FeedForward` →
`EncoderLayer`/`DecoderLayer` (atención multi-cabeza, *cross-attention* en el
decoder) → `Encoder`/`Decoder` → `PINNsformer`.

**Adaptación a este problema (escenario fijo):**
- Embedding de entrada $d_{\mathrm{in}} = 3$: $(\xi,\,\eta,\,\hat{t})$ — sólo
  coordenadas espacio-temporales (la geometría/carga es constante).
- Salida $d_{\mathrm{out}} = 2$: $(U, V)$.
- **Pseudo-secuencia temporal** (truco del paper, `utils/sampling.py`): cada
  punto se replica $k=5$ veces desplazando $\hat{t}$; la predicción real es el
  primer token.
- **Restricciones duras** de empotramiento e inicial: la salida se multiplica por
  el ansatz $\xi\,\dfrac{\hat t^2}{\hat t^2+\tau^2}$, que se anula exactamente en
  $\xi=0$ y $\hat t=0$ (con velocidad nula) y **satura a 1** para $\hat t\gg\tau$ (no
  crece con el horizonte, a diferencia del antiguo $(\hat t/\hat T)^2$). Impone esas
  condiciones por construcción; **no** modifica las capas del paper.

Hiperparámetros del paper: $k=5$, $d_{\mathrm{model}}=32$, $\mathrm{heads}=2$, $N=1$.

---

## 3. Estructura del proyecto

```
Proyecto_viga2D/
├── config.py              # Constantes, material, escenario fijo (VIGA), hiperparámetros
├── models/pinnsformer.py  # Arquitectura PINNsFormer (d_in=3, restricciones duras)
├── physics/pde_loss.py    # ElastodynamicsLoss (L_res, L_load, L_bc, L_ic, L_eq)
├── fem/fem_solver.py      # GROUND TRUTH por Elementos Finitos (scikit-fem, Newmark)
├── utils/
│   ├── sampling.py        # LHS + pseudo-secuencia temporal
│   ├── inference.py       # Inferencia vectorizada por bloques (memory-safe)
│   ├── metrics.py         # Métricas de simulación + validación + comparación PINN/FEM
│   └── registry.py        # Subida/descarga del modelo a W&B (gratis)
├── train.py               # Entrenamiento Adam + L-BFGS → checkpoints/best_model.pth
├── app.py                 # Dashboard Dash (PINN vs FEM, interactivo)
├── runpod_launch.py       # Lanza el entrenamiento en un Pod GPU de RunPod
├── Dockerfile.train       # Imagen de entrenamiento para RunPod
├── requirements.txt
└── .env.example           # Plantilla de claves (W&B, RunPod)
```

---

## 4. Instalación

```bash
pip install -r requirements.txt     # o:  uv add scikit-fem  (entorno uv)
cp .env.example .env                 # rellena tus claves
```

La solución de referencia usa **`scikit-fem`** (FEM puro en Python sobre
NumPy/SciPy, sin compilar). Es ligera y se ejecuta en local/WSL sin problema.

---

## 5. Entrenamiento

> ⚠️ **El entrenamiento NO debe ejecutarse en CPU/WSL modesto.** La
> diferenciación automática de **segundo orden** sobre miles de puntos de
> colocación agota la RAM (puede tumbar WSL). Úsalo en **GPU**.

### Opción A — RunPod (recomendado)

El flujo está pensado para **gastar el mínimo de créditos**:

1. **Construye y publica** la imagen de entrenamiento:
   ```bash
   docker build -f Dockerfile.train -t <usuario>/viga2d-train:latest .
   docker push <usuario>/viga2d-train:latest
   ```
2. Define en `.env`: `RUNPOD_API_KEY`, `WANDB_API_KEY`, `DOCKER_IMAGE`.
3. Lanza el Pod GPU; entrena, **sube el modelo a W&B y se autodestruye**:
   ```bash
   python runpod_launch.py
   ```

**GPU de alto rendimiento (techo 2.5 $/h):** `runpod_launch.py` prioriza GPUs
potentes (H100 → A100 → L40S → RTX 6000 Ada → A6000 → RTX 4090) sin superar el
techo `MAX_PRECIO_USD_H = 2.5`. La autodiferenciación de 2.º orden vuela en
H100/A100, así que un entrenamiento completo (~3000 Adam + 500 L-BFGS) tarda
**< 25 min ⇒ < 1 $** incluso en la GPU más cara. El Pod se termina
automáticamente al acabar (`finally: terminate_pod`), así que **no se acumulan
costes**. El modelo (pocos MB) queda en **W&B (almacenamiento gratuito)**.

> **Versionado en W&B:** cada entrenamiento sube una **nueva versión** del
> artifact (`v0`, `v1`, …) y mueve el alias `latest`; las versiones anteriores
> **no se borran** (siguen accesibles por su número). La app descarga `:latest`,
> así que un reentreno actualiza la visualización sin perder el histórico.

El checkpoint guarda además, para la visualización, las **métricas de
entrenamiento** (desglose final de la pérdida $\mathcal{L}_{\mathrm{res}},
\mathcal{L}_{\mathrm{bc}}, \mathcal{L}_{\mathrm{ic}}$, curva de convergencia,
nº de parámetros y épocas).

### Opción B — GPU local

```bash
python train.py --adam 3000 --lbfgs 500 --device cuda
# Prueba rápida:  python train.py --adam 500 --lbfgs 0
```

El mejor modelo se guarda en `checkpoints/best_model.pth` y, si hay
`WANDB_API_KEY`, se sube a W&B automáticamente.

---

## 6. Visualización (app Dash)

La **inferencia es ligera** (una pasada con derivadas de primer orden, evaluada
por bloques), por lo que **el dashboard SÍ se ejecuta en local/WSL** sin riesgo.

```bash
python app.py        # → http://127.0.0.1:8050
```

Al arrancar, la app busca `checkpoints/best_model.pth`; si no existe, intenta
**descargarlo de W&B** (`utils/registry.download_checkpoint`). Así puedes
entrenar en RunPod y visualizar en tu equipo sin transferir archivos a mano.

Al pulsar **«Simular y comparar»** la app evalúa la **PINN** y resuelve el **FEM**
de referencia (cacheado) sobre la misma malla e instantes, y los enfrenta.

**Interfaz:**
- **Escenario fijo** (panel lateral): tabla con $L$, $c$, $P_{\max}$,
  $\hat{t}_{\mathrm{ramp}}$, $\hat T$ y material — todo constante.
- **Métricas de entrenamiento:** semáforo de calidad, desglose de pérdidas
  ($\mathcal L_{res}, \mathcal L_{load}, \mathcal L_{bc}, \mathcal L_{ic}, \mathcal L_{eq}$),
  curva de convergencia (log) y datos del modelo.
- **Mapa animado:** malla deformada coloreada por **Von Mises**, **conmutable
  PINN/FEM** (misma escala de color para comparar), con Play/Pause y slider.
- **Oscilación $v(L,0,t)$:** curvas **PINN y FEM superpuestas**.
- **$\sigma_{xx}(x)$:** perfiles **PINN y FEM superpuestos** en el instante elegido.
- **🏆 PINN vs FEM (ground truth):** tabla de **ganador por categoría** (tiempo de
  cómputo, flecha, $\sigma_{VM}$ máx., ubicación de la tensión máxima, fidelidad,
  consulta malla-free) + **concordancia de campos** (error L2 de desplazamiento,
  $\sigma_{VM}$ y $\sigma_{xx}$ de la PINN medida contra el FEM).
- **Métricas de simulación (PINN)** y **Validación física**: *sanity-checks*,
  contraste con **Euler-Bernoulli** y, conmutable, residuo de la EDP en malla nueva.

### El *ground truth*: PINN vs Elementos Finitos

A diferencia de una PINN pura (donde "la verdad son las ecuaciones"), aquí hay un
**patrón objetivo**: la solución FEM (`fem/fem_solver.py`, `scikit-fem` + Newmark)
del **mismo** problema. La app cuantifica cuánto se acerca la PINN al FEM y resume
las **ventajas de cada método** (el FEM es la referencia física; la PINN es
malla-free, da derivadas exactas por autograd y responde una consulta puntual sin
re-mallar). Como complemento se mantiene el contraste analítico con
**Euler-Bernoulli** ([docs/03 §12](docs/03_Respuestas_dudas.md)).

**Validación conmutable** (`config.py → VALIDACION`):
- `fem_ground_truth` (bool) y `fem_substeps`: solución FEM de referencia.
- `referencia_analitica` (bool): contraste Euler-Bernoulli (fórmula cerrada).
- `residuo_en_malla` (bool) y `n_residuo_check`: residuo de la EDP en malla nueva
  (autograd de 2.º orden sobre un subconjunto pequeño de puntos).

---

## 7. ¿Por qué este reparto RunPod + W&B + local?

| Tarea | Dónde | Motivo |
|---|---|---|
| Entrenamiento (2.º orden, pesado) | **RunPod GPU** | Imposible en WSL; barato y efímero |
| Almacenamiento del modelo | **W&B (gratis)** | Apagar el Pod ⇒ no pagar disco |
| Visualización (1.er orden, ligera) | **Local / WSL** | Una app Dash 24/7 en RunPod gastaría créditos sin aportar nada |

---

## 8. Decisiones de diseño (versión simplificada)

- **Escenario fijo** (`config.VIGA`): geometría, carga y material constantes; la
  red pasa de $d_{in}=7$ a $d_{in}=3$. Simplifica el problema sin tocar el núcleo.
- **Ground truth por FEM** (`fem/fem_solver.py`): valida la PINN contra una
  solución objetiva (no sólo contra el residuo), en la misma malla e instantes.
- **Carga en rampa lineal**: perfil temporal de la BC más interpretable que la
  sigmoide; arranca en 0 (coherente con el reposo). No cambia la EDP.
- **Régimen de flexión** (ver arriba y [docs/05](docs/05_Diagnostico_resultados.md)):
  $\hat T=100$ (cubre $>1$ periodo de flexión) y $\sigma_{\mathrm{scale}}$ = tensión de
  flexión, para que la PINN, el FEM y Euler-Bernoulli **triangulen**.
- **Restricciones duras** (ansatz saturante $\xi\,\hat t^2/(\hat t^2+\tau^2)$) +
  **peso propio de la carga** ($w_{load}$) + **reescalado del residuo** ($\times\hat c$):
  primeras defensas contra el colapso a la solución trivial $u=v=0$.
- **Equilibrio seccional** (`L_eq`, **lo que de verdad rompe el colapso**): el problema
  es de flexión *controlada por fuerza* y la EDP del interior es **homogénea**
  ($\rho\ddot u=\nabla\!\cdot\sigma$) → $u\approx0$ la satisface con residuo $\approx0$,
  así que la red colapsa. Se impone el **equilibrio integral** que el residuo local no
  transmite: en toda sección $\xi$, cortante $\int\hat\tau_{xy}\,d\eta=-0.02\,g$ y
  momento $\int\hat\sigma_{xx}\,\eta\,d\eta=+0.3(1-\xi)g$ (consecuencias exactas de
  $\nabla\!\cdot\sigma=0$, **no datos**; verificadas contra el FEM). Con $u\approx0$ el
  momento da 0 → penalización fuerte → la red sale del trivial.
- **Adimensionalización completa**: sin ella el residuo se escala por $\sim10^{11}$
  y la PINN no converge.
- **Inferencia por bloques** (`chunk`) y **doble optimizador** Adam → L-BFGS.
- **Validación multinivel** (`utils/metrics.py`): comparación PINN/FEM, métricas de
  simulación, *sanity-checks* y contraste con Euler-Bernoulli. **El termómetro de
  calidad es el error L2 PINN vs FEM**, no la pérdida total (que puede ser baja en un
  mínimo trivial degenerado).

---

## 9. Referencias

- Z. Zhao, X. Ding, B. A. Prakash. *PINNsFormer: A Transformer-Based Framework
  for Physics-Informed Neural Networks.* ICLR 2024.
- Raissi, Perdikaris, Karniadakis. *Physics-Informed Neural Networks.* JCP 2019.
