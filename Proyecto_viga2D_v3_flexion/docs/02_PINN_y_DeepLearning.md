# 02 — Cómo lo resuelve la red: PINN + Deep Learning

> Objetivo: entender **cómo** una red neuronal resuelve la EDP del
> [documento 01](01_Sistema_y_Elasticidad.md): qué son las derivadas y por qué
> se calculan, por qué hace falta 2.º orden, cómo funciona la pérdida y el
> backprop, y la arquitectura PINNsFormer al completo, con sus ventajas frente a
> Elementos Finitos.

---

## 1. La idea central de una PINN (Physics-Informed Neural Network)

Una red neuronal es un **aproximador universal de funciones**. La idea de una
PINN es brillante por simple:

> En lugar de aprender de datos (pares entrada→salida), la red **es** la solución
> de la EDP. Se la entrena para que la función que representa **satisfaga la
> ecuación física** y sus condiciones de contorno.

La red $\mathcal{N}_\theta$ (con parámetros $\theta$) toma un punto $(x, y, t)$
y devuelve el desplazamiento en ese punto:

$$\mathcal{N}_\theta(x, y, t) \;\longrightarrow\; (u, v)$$

Para que $\mathcal{N}_\theta$ sea la solución correcta debe cumplir, en **todo**
el dominio, las ecuaciones de Navier-Cauchy. Definimos el **residuo**: lo que
sobra al sustituir la predicción de la red en la EDP:

$$r_u = \mu\,(u_{xx}+u_{yy}) + (\lambda+\mu)\,\frac{\partial}{\partial x}(u_x+v_y) - \rho\,u_{tt}$$

Si $\mathcal{N}_\theta$ fuese la solución exacta, $r_u = 0$ en todas partes.
**Entrenar la PINN = ajustar $\theta$ para que $r_u \approx 0$ y $r_v \approx 0$
en miles de puntos**, además de cumplir las condiciones de contorno. **No hace
falta ningún dato de referencia: la física es la supervisión.**
(Esto se llama *unsupervised / physics-driven*.)

---

## 2. ¿Cómo se resuelve? Derivadas y autodiferenciación

### 2.1 Por qué se calculan derivadas
El residuo *es* una combinación de derivadas de la solución ($u_{xx}$, $u_{tt}$,
etc.). Para evaluar "cuánto incumple la red la EDP" en un punto, hay que calcular
esas derivadas de la salida de la red respecto a sus **entradas** $(x, y, t)$.

### 2.2 Cómo: autodiferenciación (autograd), no diferencias finitas
Aquí está la magia que distingue a las PINN de los métodos clásicos. La red
neuronal es una **composición de funciones diferenciables** (lineales + la
activación WaveAct). Por tanto su derivada respecto a las entradas se puede
calcular **exactamente**, hasta precisión de máquina, con la **regla de la
cadena automática** (autograd de PyTorch). No hay malla, no hay aproximación por
diferencias finitas, no hay error de discretización espacial.

En el código, [`_grad`](../physics/pde_loss.py#L35-L41):

```python
def _grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph=True, create_graph=True)[0]
```

El detalle **crítico** es `create_graph=True`: le dice a PyTorch que la propia
operación "calcular la derivada" quede registrada en el grafo de cómputo, para
poder **derivar otra vez** (derivada de la derivada) y, después, derivar todo
respecto a los pesos $\theta$. Sin esto no se podrían obtener las derivadas segundas.

### 2.3 Por qué derivadas de SEGUNDO orden (y por qué importa tanto)
La ecuación de Navier-Cauchy contiene $u_{xx}$, $u_{yy}$, $u_{tt}$… → **derivadas
segundas**. La red se evalúa así, en cadena (ver
[`loss_residual`](../physics/pde_loss.py#L102-L115)):

$$U \xrightarrow{\text{autograd \#1}} U_\xi \xrightarrow{\text{autograd \#2}} U_{\xi\xi}$$

$$U \xrightarrow{\text{autograd \#1}} U_{\hat t} \xrightarrow{\text{autograd \#2}} U_{\hat t\hat t}$$

Es decir, se **deriva la red dos veces** respecto a sus entradas solo para
*construir* el residuo. Y aquí viene el punto que mencionas en tu pregunta:

### 2.4 La conexión con el backpropagation (esto es lo profundo)
Para entrenar, hay que minimizar la pérdida $\mathcal{L}(\theta)$ (que contiene
el residuo) ajustando los pesos $\theta$. Eso requiere
$\partial\mathcal{L}/\partial\theta$ — el **backpropagation** habitual. Pero
$\mathcal{L}$ ya contiene **derivadas segundas de la red respecto a sus entradas**.
Por tanto:

> El gradiente de la pérdida respecto a los pesos es una **derivada de una
> derivada segunda** → en la práctica PyTorch construye un grafo de **tercer
> orden** de diferenciación. Esto es lo que hace el entrenamiento de PINNs
> **caro en memoria y cómputo**.

Concretamente: el grafo de $U_{\xi\xi}$, que ya es grande, debe a su vez ser
recorrido hacia atrás para obtener $\partial\mathcal{L}/\partial\theta$. Por eso
el [README](../README.md) avisa de que **el entrenamiento no se puede hacer en
CPU/WSL modesto** (agota la RAM) y se lanza en GPU, y por eso
[train.py](../train.py#L97-L114) usa **acumulación de gradiente por bloques**
(`chunk`): trocea los miles de puntos para que el enorme grafo de 2.º orden quepa
en la VRAM, obteniendo exactamente el mismo gradiente medio.

> **Resumen para defensa:** "El residuo de la EDP exige derivar la red dos veces
> respecto a las coordenadas (vía autograd con `create_graph=True`). Como esas
> derivadas forman parte de la función de pérdida, el backprop respecto a los
> pesos diferencia *a través* de ellas: es diferenciación de orden superior, de
> ahí el alto coste de memoria que obliga a entrenar en GPU y por bloques."

---

## 3. La función de pérdida

La pérdida total tiene **cinco** componentes ([`total`](../physics/pde_loss.py)):

$$\mathcal{L} = w_{\mathrm{res}}\mathcal{L}_{\mathrm{res}} + w_{\mathrm{load}}\mathcal{L}_{\mathrm{load}} + w_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}} + w_{\mathrm{ic}}\mathcal{L}_{\mathrm{ic}} + w_{\mathrm{eq}}\mathcal{L}_{\mathrm{eq}}$$

| Componente | Qué penaliza | Detalle |
|---|---|---|
| $\mathcal{L}_{\mathrm{res}}$ | Incumplir la EDP en el interior | media de $r_u^2 + r_v^2$ (reescalado $\times\hat c$ para equilibrar su magnitud) |
| $\mathcal{L}_{\mathrm{load}}$ | Extremo cargado $\xi=1$ | $\sigma_{xx}=0$, $\tau_{xy}=\tau_{\mathrm{app}}$ (condición **motora**) |
| $\mathcal{L}_{\mathrm{bc}}$ | Contorno homogéneo | empotramiento ($u=v=0$) + superficies libres ($\sigma_{yy}=\tau_{xy}=0$) |
| $\mathcal{L}_{\mathrm{ic}}$ | Reposo inicial | $u=v=0$ **y** $u_t=v_t=0$ en $t=0$ (monitor: ya lo impone la restricción dura) |
| $\mathcal{L}_{\mathrm{eq}}$ | **Equilibrio seccional** | cortante/momento integral por sección; **rompe el colapso trivial** (ver doc 04 §12) |

- Todas son **errores cuadráticos medios (MSE)** de cantidades que deberían ser
  cero. La solución exacta daría $\mathcal{L}=0$.
- **Pesos $w$:** $w_{\mathrm{res}}=1$, $w_{\mathrm{load}}=20$, $w_{\mathrm{bc}}=5$,
  $w_{\mathrm{ic}}=1$, $w_{\mathrm{eq}}=20$ ([config.py](../config.py)). La **carga** y el
  **equilibrio** llevan peso alto porque son las condiciones que alejan del "mínimo
  tramposo" $u=v=0$ (que cumple el residuo homogéneo pero ignora la flexión). Equilibrar
  estos pesos es uno de los puntos finos —y dificultades conocidas— de las PINN.

### Puntos de colocación (¿dónde se evalúa la pérdida?)
No hay malla. Se muestrean puntos **aleatorios** del dominio con **Latin
Hypercube Sampling** (LHS, [sampling.py](../utils/sampling.py#L24-L36)), que
cubre el espacio de forma más uniforme que el muestreo puramente aleatorio.
Cantidades ([config.py](../config.py)): 6000 interiores, 1000 por frontera, 1500
iniciales y 400 **secciones** ($\times17$ nodos en $\eta$) para el equilibrio seccional.
Son los puntos donde se exige que se cumpla la física.

---

## 4. Adimensionalización: por qué y cómo

**El problema:** el acero tiene $E \approx 2.1\cdot10^{11}$ Pa y los desplazamientos son del
orden de **micras** ($10^{-6}$ m). Si la red trabajara en unidades del SI:
- las salidas que debe aprender son $\approx 10^{-6}$ (gradientes minúsculos),
- el residuo de la EDP queda escalado por factores $\approx 10^{11}$,
- las distintas componentes de la pérdida tienen escalas dispares por órdenes de
  magnitud → el optimizador no puede equilibrarlas y **la red no converge**.

Las redes neuronales entrenan bien cuando entradas, salidas y residuos son de
**orden unidad (~1)**. Por eso *todo* el problema se reescribe en variables
adimensionales ([config.py](../config.py#L10-L43)):

$$\xi = \frac{x}{L} \in [0,1], \quad \eta = \frac{y}{c} \in [-1,1], \quad \hat{t} = \frac{t}{T_{\mathrm{ref}}}$$

$$U = \frac{u}{U_{\mathrm{ref}}}, \quad V = \frac{v}{U_{\mathrm{ref}}}, \qquad T_{\mathrm{ref}} = L_{\mathrm{ref}}\sqrt{\frac{\rho}{E}}, \quad \sigma_{\mathrm{scale}} = \frac{P_{\mathrm{ref}}\,L_{\mathrm{ref}}^2}{2\,c_{\mathrm{ref}}^3}, \quad U_{\mathrm{ref}} = \frac{L_{\mathrm{ref}}\,\sigma_{\mathrm{scale}}}{E}$$

> **No basta con que sean adimensionales: hay que acertar la ESCALA** (ver
> [doc 05](05_Diagnostico_resultados.md)). $\sigma_{\mathrm{scale}}$ es la **tensión de
> flexión** $Mc/I$ (no la tracción aplicada $3P/4c$): así el desplazamiento adimensional
> es O(1) y no $\sim(L/c)^2\!\approx\!225$. Y $T_{\mathrm{ref}}$ es el tiempo de **onda**,
> $\sim50\times$ más rápido que la flexión, por lo que el horizonte debe ser
> $\hat T=100$ (cubre $>1$ periodo de flexión), no $\hat T=4$ (sólo ondas).

Sustituyendo en Navier-Cauchy, **los coeficientes quedan de orden 1**:
$\hat{\mu} = 1/(2(1+\nu)) \approx 0.385$, $\hat{\lambda} = \nu/(1-\nu^2) \approx 0.33$.
El residuo adimensional que calcula el código ([pde_loss.py](../physics/pde_loss.py#L120-L123)):

$$r_U = \hat\mu\!\left(\frac{U_{\xi\xi}}{\hat L^2}+\frac{U_{\eta\eta}}{\hat c^2}\right) + (\hat\lambda+\hat\mu)\!\left(\frac{U_{\xi\xi}}{\hat L^2}+\frac{V_{\xi\eta}}{\hat L\,\hat c}\right) - U_{\hat t\hat t} = 0$$

Ventaja extra: como las ecuaciones adimensionales **solo dependen de $\nu$** (no de
$E$ ni $\rho$), una red entrenada con acero sirve para el aluminio (Poisson casi
igual); el material físico se recupera **reescalando** la salida con $U_{\mathrm{ref}}$,
$T_{\mathrm{ref}}$, $\sigma_{\mathrm{scale}}$ de cada material en
[inference.py](../utils/inference.py#L115-L122). **No se reentrena por material.**

---

## 5. La arquitectura PINNsFormer (deep learning, en detalle)

PINNsFormer (Zhao et al., ICLR 2024) es un **Transformer** adaptado a PINNs.
Implementación: [models/pinnsformer.py](../models/pinnsformer.py). La motivación
del paper: los MLP estándar tienen dificultades con la **dependencia temporal**;
un Transformer, que procesa **secuencias**, captura mejor la evolución en el
tiempo.

### 5.1 Entradas y salidas de la red

> **Versión simplificada (escenario fijo).** Tras el feedback de los tutores, la
> red ya **no** recibe parámetros: la geometría/carga es constante (`config.VIGA`)
> y la entrada es $d_{\mathrm{in}}=3$. La §6, que describe la red *paramétrica*
> original ($d_{\mathrm{in}}=7$), se conserva como discusión del compromiso
> simplicidad ↔ generalización (clave para defender la decisión de diseño).

**Entrada** ($d_{\mathrm{in}} = 3$): $(\xi,\; \eta,\; \hat{t})$ — coordenadas
adimensionales (dónde y cuándo). Respecto a ellas se derivan los residuos.

**Salida** ($d_{\mathrm{out}} = 2$): $(U, V)$ — desplazamientos adimensionales.

- Las 3 entradas se **normalizan a $\approx[-1,1]$** con buffers $(lo, hi)$
  ([pinnsformer.py](../models/pinnsformer.py#L164-L177)) — misma lógica de
  acondicionamiento que la adimensionalización, ahora a nivel de red.
- **Restricción dura**: la salida se multiplica por el ansatz
  $\xi\,\hat t^2/(\hat t^2+\tau^2)$, que se anula exactamente en el empotramiento
  ($\xi=0$) y en el instante inicial ($\hat t=0$, con velocidad nula) y **satura a 1**
  para $\hat t\gg\tau$ → impone esas condiciones por construcción sin que la envolvente
  crezca con el horizonte (el antiguo $(\hat t/\hat T)^2$ crecía $\times\hat T^2$ y
  descondicionaba). Por sí sola no basta para evitar el colapso trivial: ver el
  **equilibrio seccional** (§3 y doc 04 §12).

### 5.2 La pseudo-secuencia temporal (el truco del paper)
Un Transformer procesa secuencias, pero aquí cada muestra es un punto, no una
secuencia. Solución de PINNsFormer
([sampling.py](../utils/sampling.py#L39-L50)): **replicar cada punto $k=5$ veces
desplazando solo $\hat{t}$** un paso minúsculo $\Delta = \text{seq\_step} = 10^{-3}$:

$$(\xi,\,\eta,\,\hat{t}) \;\longrightarrow\; \big[(\xi,\,\eta,\,\hat{t}),\;(\xi,\,\eta,\,\hat{t}+\Delta),\;(\xi,\,\eta,\,\hat{t}+2\Delta),\;\ldots,\;(\xi,\,\eta,\,\hat{t}+4\Delta)\big]$$

con forma $(N,3) \to (N,\,k{=}5,\,3)$. Así cada punto se convierte en una
mini-secuencia temporal que el Transformer procesa, capturando la
**coherencia temporal local**. La predicción "real" es el **primer token**
(índice 0); los demás solo refuerzan la coherencia en la pérdida.

### 5.3 El flujo interno (secuencia de capas)

```
(ξ, η, t̂)
   → normalizar a [-1, 1]                       (buffers lo, hi)
   → Linear embedding   (3 → d_model = 32)
   → ENCODER  (N=1 capa)
        self-attention (heads=2) + FeedForward   con WaveAct y conexiones residuales
   → DECODER  (N=1 capa)
        cross-attention sobre salida del encoder + FeedForward
   → Cabeza MLP de salida   (32 → 512 → 512 → 2)  con WaveAct
   → (U, V)
```

Componentes, fieles al repo oficial:

- **WaveAct** ([L41-L50](../models/pinnsformer.py#L41-L50)) — la activación
  estrella:

$$f(x) = w_1\sin(x) + w_2\cos(x), \quad w_1, w_2 \in \mathbb{R} \text{ aprendibles}$$

  En vez de ReLU/tanh, usa ondas seno/coseno. ¿Por qué? Las soluciones de problemas
  elastodinámicos son **ondas/oscilaciones**; una activación senoidal representa
  estas funciones periódicas con muchísima más facilidad (sesgo inductivo
  adecuado) y, al ser suave ($C^\infty$), da derivadas limpias para el residuo.

- **Multi-Head Attention** ($\mathrm{heads}=2$): el mecanismo que permite que los tokens
  de la pseudo-secuencia "se miren" entre sí y compartan información temporal.
- **Encoder/Decoder con conexiones residuales** ($x \leftarrow x + \mathrm{Attn}(x)$,
  $x \leftarrow x + \mathrm{FF}(x)$): estructura clásica de Transformer; el decoder usa
  **cross-attention** sobre la memoria del encoder.
- **Cabeza MLP** ancha (512) que proyecta del espacio del Transformer a $(U,V)$.

Hiperparámetros (del paper, [config.py](../config.py#L142-L153)):
$d_{\mathrm{model}}=32$, $\mathrm{heads}=2$, $N=1$, $k=5$.
Es una red **pequeña** — la potencia no viene del tamaño sino de la estructura
(atención + WaveAct) y de la información física.

### 5.4 Por qué las entradas vienen separadas
`forward(self, xi, eta, t, params)` recibe las coordenadas **separadas**
([L174](../models/pinnsformer.py#L174)) en lugar de un único tensor. Razón
técnica: para que `torch.autograd.grad` pueda derivar la salida respecto a $\xi$,
$\eta$, $\hat{t}$ **individualmente** (cada una es una "hoja" derivable del grafo),
que es lo que exige construir el residuo. Por dentro se concatenan.

---

## 6. Sobre qué generaliza la red (diseño paramétrico — opcional)

> ⚠️ **Esta sección describe la versión PARAMÉTRICA original ($d_{in}=7$), hoy
> desactivada** en favor del escenario fijo (más simple). Se conserva porque
> explica el **compromiso de diseño** (simplicidad ↔ generalización) y cómo
> recuperar la generalización si se desea: basta volver a meter $(\hat L,\hat c,
> \hat p,\hat t_{\mathrm{imp}})$ como entradas y muestrearlos por LHS.

Una PINN clásica resuelve **un** problema concreto: una geometría, una carga.
Cambiar $L$ o $P_{\max}$ obligaría a **reentrenar** desde cero. La versión
paramétrica daba el salto a una **familia de soluciones**:

> Al meter $(\hat L, \hat c, \hat p, \hat t_{\mathrm{imp}})$ como **entradas adicionales** de la
> red, esta no aprende *una* solución sino una **familia de soluciones**
> parametrizada por la geometría y la carga.

La red generaliza sobre, simultáneamente:
- $\hat L \in [1, 2]$ — longitud de viga (1–2 m),
- $\hat c \in [0.05, 0.15]$ — semi-altura (5–15 cm),
- $\hat p \in [0.5, 1.5]$ — amplitud de la carga,
- $\hat t_{\mathrm{imp}} \in [0.5, 1.5]$ — instante del impacto,
- y, vía reescalado, sobre el **material** (acero/aluminio).

**Consecuencia práctica (lo que se ve en el dashboard):** mover los sliders de
$L$, $c$, $P_{\max}$, $t_{\mathrm{imp}}$ o cambiar de material da resultados **al instante
(<500 ms)** porque solo hay que *evaluar* la red, **no reentrenar**. Esto es lo
que el [README §8](../README.md) destaca como mejora clave: $\hat t_{\mathrm{imp}}$
*tenía* que ser entrada de la red para poder variarse desde la app.

---

## 7. Entrenamiento: Adam + L-BFGS

Estrategia estándar en PINN de alta precisión ([train.py](../train.py#L159-L201)):

1. **Fase A — Adam** (3000–5000 épocas, $\eta=10^{-3}$): optimizador robusto que
   explora rápido el paisaje de pérdida y se acerca a una buena región.
2. **Fase B — L-BFGS** (200–500 pasos, line search *strong_wolfe*): optimizador
   de segundo orden (cuasi-Newton) que **refina con altísima precisión** una vez
   cerca del mínimo. Las PINN suelen necesitar este pulido final para bajar el
   residuo varios órdenes de magnitud.

Detalles de robustez: acumulación de gradiente por `chunk` (memoria acotada, §2.4),
guardado del **mejor** modelo, y parada controlada por tiempo/`SIGTERM` para no
perder progreso si RunPod corta el Pod.

---

## 8. Ventajas frente a Elementos Finitos (FEM) y métodos clásicos

| Aspecto | FEM / Diferencias finitas | Esta PINN |
|---|---|---|
| **Malla** | Necesita **mallar** el dominio (costoso, sobre todo 3D / geometrías complejas). | **Sin malla** (*meshless*): solo puntos de colocación muestreados. |
| **Derivadas** | Aproximadas por la discretización (error de truncamiento). | **Exactas** por autodiferenciación. |
| **Solución** | Discreta (valores en nodos); interpolar entre nodos. | **Continua y diferenciable** en todo el dominio: se evalúa en cualquier $(x,y,t)$. |
| **Cambiar parámetros** ($L$, carga…) | Hay que **re-mallar y re-resolver** cada caso. | Una red paramétrica resuelve **toda la familia**; evaluación instantánea (tiempo real en la app). |
| **Problemas inversos / datos** | Difícil incorporar mediciones dispersas. | Natural: basta añadir un término de datos a la pérdida. |
| **Alta dimensión** | Sufre la "maldición de la dimensionalidad" (nº de nodos explota). | Escala mucho mejor en dimensiones altas. |

**Honestidad para la defensa (debilidades conocidas):** para *un solo* caso 2D
bien definido, un FEM maduro suele ser **más rápido y más preciso** que una PINN,
y el entrenamiento PINN es delicado (equilibrio de pesos de la pérdida,
convergencia). El valor de la PINN aquí **no** es competir con FEM en un caso
aislado, sino: (1) la **parametrización** —una red para infinitas vigas, con
inferencia en tiempo real—, (2) ser **meshless** y **diferenciable**, y (3) la
extensibilidad natural a problemas inversos. Saber decir esto demuestra criterio.

---

### ✅ Checklist de dominio
- [ ] Idea de PINN: la red *es* la solución; la física es la supervisión (sin datos).
- [ ] Qué es autodiferenciación y por qué da derivadas exactas (vs diferencias finitas).
- [ ] Por qué hay 2.º orden y cómo el backprop diferencia *a través* de esas derivadas (coste de memoria → GPU + chunks).
- [ ] Las cinco componentes de la pérdida y por qué la **carga** y el **equilibrio seccional** llevan peso alto (evitar la solución trivial).
- [ ] Por qué se adimensionaliza (residuo $\sim 10^{11}$, salidas $\sim 10^{-6}$ → no converge).
- [ ] WaveAct y por qué seno/coseno encajan con un problema de ondas.
- [ ] La pseudo-secuencia temporal y el primer-token-como-predicción.
- [ ] Sobre qué generaliza la red paramétrica y por qué eso da inferencia en tiempo real.
- [ ] Ventajas y debilidades honestas frente a FEM.
