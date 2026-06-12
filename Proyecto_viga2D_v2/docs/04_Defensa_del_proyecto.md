# 04 — Defensa del proyecto (explicación integral)

> Documento pensado como **guion de defensa** ante tribunal experto en deep
> learning y matemáticas. Explica el proyecto entero —problema, matemáticas,
> método, arquitectura, infraestructura y resultados— con rigor pero a nivel de
> estudiante de máster: completo y honesto, sin academicismo innecesario. Para el
> detalle fino, se enlaza a los docs [01](01_Sistema_y_Elasticidad.md),
> [02](02_PINN_y_DeepLearning.md) y [03](03_Respuestas_dudas.md).

---

## 1. En una frase

Resuelvo la **vibración de una viga empotrada 2D golpeada por un impacto** con una
**red neuronal informada por la física** (PINN), usando la arquitectura
**PINNsFormer** (un Transformer, Zhao et al., ICLR 2024), entrenada para ser
**paramétrica** —una sola red sirve para distintas geometrías y cargas— y servida
en un **dashboard interactivo** con todo el flujo MLOps (entreno en GPU cloud,
modelo en W&B, visualización local).

---

## 2. El problema físico

Una viga de longitud $L$ y semi-altura $c$ está **empotrada** en $x=0$ (pared) y
**libre** en $x=L$. En el extremo libre recibe una **carga cortante de impacto**:
un tirón transversal que aparece de forma suave (rampa sigmoidea) en un instante
$t_{\mathrm{imp}}$. Queremos saber **cómo se desplaza y se tensiona la viga en el
tiempo**: la oscilación transitoria y dónde corre más riesgo de fallar.

Es un problema de **elastodinámica lineal**: material elástico (recupera su forma),
con **inercia** (hay ondas y vibración) y **pequeñas deformaciones** (relación
tensión–deformación lineal). Ver [doc 01 §4](01_Sistema_y_Elasticidad.md).

---

## 3. El modelo matemático: de Newton a Navier-Cauchy

La incógnita es el **campo de desplazamientos** $\vec d(x,y,t)=(u,v)$: a cada punto
material y cada instante le asigna cuánto se ha movido. La cadena es:

$$\text{desplazamiento } (u,v) \;\to\; \text{deformación } \varepsilon \;\to\;
  \text{tensión } \sigma \;\to\; \text{equilibrio (Newton)}$$

- **Deformaciones** (derivadas 1.ª del desplazamiento):
  $\varepsilon_{xx}=\partial_x u,\;\varepsilon_{yy}=\partial_y v,\;
   \varepsilon_{xy}=\tfrac12(\partial_y u+\partial_x v)$.
- **Tensiones** (ley de Hooke, tensión plana):
  $\sigma_{xx}=(\lambda+2\mu)\varepsilon_{xx}+\lambda\varepsilon_{yy}$, etc.
- **Equilibrio** = 2.ª ley de Newton sobre cada trocito → **ecuaciones de
  Navier-Cauchy**:

$$\rho\,\partial_{tt}u = \mu(\partial_{xx}u+\partial_{yy}u)
   + (\lambda+\mu)\,\partial_x(\partial_x u+\partial_y v)$$

(y la análoga para $v$). Es un **sistema de EDPs** acopladas (derivadas en $x,y$ **y**
$t$); el acoplamiento entre $u$ y $v$ es el **efecto Poisson** vía $\lambda$.

**Por qué 2D en tensión plana:** una viga es delgada en $z$ con caras libres, luego
$\sigma_{zz}=\sigma_{xz}=\sigma_{yz}=0$ → de ahí $\lambda=E\nu/(1-\nu^2)$. Un modelo
1D clásico (Euler-Bernoulli, una ODE) solo aproximaría: desprecia el cortante.

**Condiciones** (cierran el problema): IC de reposo ($u=v=0,\ \dot u=\dot v=0$ en
$t=0$); empotramiento ($u=v=0$ en $x=0$); superficies libres
($\sigma_{yy}=\tau_{xy}=0$ en $y=\pm c$); extremo cargado ($\sigma_{xx}=0$,
$\tau_{xy}=\tau_{\mathrm{app}}$ en $x=L$), con perfil parabólico $(1-\eta^2)$ y rampa
$\mathrm{sigmoid}(\alpha(\hat t-\hat t_{\mathrm{imp}}))$.

---

## 4. Por qué una PINN (y no FEM ni datos)

Una **PINN** invierte la lógica del deep learning supervisado: la red **es** la
solución $\mathcal N_\theta(x,y,t)\to(u,v)$, y la **pérdida es el residuo de la
física**, no un error contra etiquetas. No hay datos de referencia: se minimiza
cuánto incumplen las ecuaciones la red y sus condiciones.

$$\mathcal L = w_{\mathrm{res}}\underbrace{\lVert r_u\rVert^2+\lVert r_v\rVert^2}_{\text{residuo EDP}}
   + w_{\mathrm{bc}}\,\mathcal L_{\mathrm{bc}} + w_{\mathrm{ic}}\,\mathcal L_{\mathrm{ic}}$$

donde $r_u=\rho\,\partial_{tt}u-(\dots)$ es el residuo: cuánto se aparta la red de
cumplir Navier-Cauchy en cada punto. **Ventajas frente a FEM:** malla-libre,
solución continua y derivable, y —clave aquí— **parametrizable** (la red recibe
geometría y carga como entradas y generaliza sin re-resolver). El precio: el
entrenamiento es un problema de optimización no convexo, delicado (ver §11).

**Autodiferenciación de 2.º orden:** el residuo necesita $\partial_{tt}$,
$\partial_{xx}$… de la red respecto a sus entradas. Se obtienen con `autograd` de
PyTorch (`create_graph=True`), derivando dos veces a través de la red. Es la
operación cara del proyecto y la razón de entrenar en GPU. Ver
[doc 02](02_PINN_y_DeepLearning.md).

---

## 5. Adimensionalización (imprescindible)

Con unidades del SI el problema es **inentrenable**: $E\sim10^{11}$ Pa,
desplazamientos $\sim10^{-6}$ m → los residuos se escalan por $\sim10^{11}$. Se
trabaja en variables adimensionales:

$$\xi=\frac xL\in[0,1],\quad \eta=\frac yc\in[-1,1],\quad \hat t=\frac t{T_{\mathrm{ref}}},
  \quad U=\frac u{U_{\mathrm{ref}}}$$

con $T_{\mathrm{ref}}=L_{\mathrm{ref}}\sqrt{\rho/E}$ (tiempo de onda elástica),
$\sigma_{\mathrm{scale}}=3P_{\mathrm{ref}}/(4c_{\mathrm{ref}})$ y
$U_{\mathrm{ref}}=L_{\mathrm{ref}}\sigma_{\mathrm{scale}}/E$. Así los coeficientes de
la EDP quedan de orden unidad ($\hat\mu\approx0.385$, $\hat\lambda\approx0.33$). El
selector de material de la app reescala la solución adimensional a unidades físicas
con $U_{\mathrm{ref}},T_{\mathrm{ref}}$ de cada material (acero/aluminio comparten
$\nu$ casi idéntico, así que la solución adimensional vale para ambos).

---

## 6. La arquitectura PINNsFormer

Implementación fiel del repo oficial (`models/pinnsformer.py`):

- **WaveAct**: activación de onda $f(x)=w_1\sin x+w_2\cos x$, con $w_1,w_2$
  aprendibles. Sustituye a `tanh`/`ReLU`; al ser periódica capta mejor las
  oscilaciones de un problema dinámico.
- **Encoder–Decoder Transformer**: *self-attention* en el encoder, *cross-attention*
  en el decoder, conexiones residuales. Hiperparámetros del paper: $d_{\mathrm{model}}=32$,
  $\mathrm{heads}=2$, $N=1$.
- **Pseudo-secuencia temporal** (el truco del paper): cada punto $(\xi,\eta,\hat t)$
  se replica $k=5$ veces desplazando $\hat t$ un paso $\Delta$; el Transformer procesa
  esa mini-secuencia y la predicción real es el **primer token**. Convierte un
  problema puntual en uno secuencial, que es donde el Transformer brilla.

**Extensión propia (PINN paramétrica):** la entrada no es de dimensión 2 sino
**7**: $(\xi,\eta,\hat t,\hat L,\hat c,\hat p,\hat t_{\mathrm{imp}})$. Las 3 primeras
son *coordenadas* (dónde/cuándo evalúo); las 4 últimas son *parámetros* (qué viga /
qué carga). Salida: $(U,V)$. Durante el entrenamiento los 4 parámetros se **varían**
(muestreo Latin Hypercube en sus rangos), de modo que la red aprende la **familia
completa** de soluciones; en inferencia se **fijan** a los sliders. Ver
[doc 03 §9](03_Respuestas_dudas.md).

---

## 7. La función de pérdida en detalle

`physics/pde_loss.py` calcula tres bloques sobre conjuntos de puntos distintos:

- $\mathcal L_{\mathrm{res}}$: residuo de Navier-Cauchy en el **interior** (usa
  derivadas 2.ª).
- $\mathcal L_{\mathrm{bc}}$: empotramiento + superficies libres + extremo cargado
  (las tres últimas vía tensiones, derivadas 1.ª).
- $\mathcal L_{\mathrm{ic}}$: reposo en $t=0$ (desplazamiento y velocidad nulos).

Pesos actuales: $w_{\mathrm{res}}=1,\ w_{\mathrm{bc}}=w_{\mathrm{ic}}=10$. El
balance entre estos pesos es **crítico**: si las condiciones homogéneas pesan
demasiado, la red colapsa a $u=v=0$ (ver §11).

---

## 8. Entrenamiento

- **Muestreo**: Latin Hypercube (mejor cobertura que aleatorio puro) del dominio
  paramétrico; 4000 puntos de residuo, 1000/frontera, 1500 de IC.
- **Dos fases**: **Adam** (exploración del paisaje de pérdida) → **L-BFGS** con
  *strong Wolfe* (refinado de alta precisión). Estándar en PINNs.
- **Acumulación de gradiente por bloques** (`chunk`): el grafo de 2.º orden a través
  del Transformer agota la VRAM; se procesa en sub-lotes ponderando por su fracción,
  obteniendo el mismo gradiente medio con memoria acotada.
- **Robustez**: presupuesto de tiempo y captura de `SIGTERM` → siempre guarda y
  sube el mejor modelo aunque RunPod corte el Pod.

---

## 9. Infraestructura (MLOps)

Reparto pensado para **minimizar coste** sin sacrificar capacidad:

| Tarea | Dónde | Por qué |
|---|---|---|
| Entrenamiento (2.º orden, pesado) | **RunPod GPU** (H100/A100, techo 2.5 $/h) | Inviable en CPU/WSL; potente y efímero |
| Almacenamiento del modelo | **W&B** (gratis, versionado) | Apagar el Pod ⇒ no pagar disco |
| Visualización (1.er orden, ligera) | **Local/WSL** | Una app 24/7 en cloud gastaría créditos |

El Pod entrena, sube el checkpoint a W&B (artifact versionado, alias `latest`) y se
**autodestruye** (`finally: terminate_pod`). La app local descarga `:latest` si no
hay fichero local. Coste típico de un entrenamiento completo: **< 1 $**.

---

## 10. La aplicación de visualización

Dash + Plotly, tema oscuro. La inferencia es ligera (1.er orden, por bloques), así
que corre en local. Paneles:

- **Control**: material, $L$, $c$, $P_{\max}$, $\hat t_{\mathrm{imp}}$, botón Simular.
- **Métricas de entrenamiento**: semáforo de calidad, pérdidas finales
  ($\mathcal L_{\mathrm{res}},\mathcal L_{\mathrm{bc}},\mathcal L_{\mathrm{ic}}$),
  curva de convergencia, nº de parámetros y épocas (leídas del checkpoint).
- **Malla deformada animada** coloreada por **Von Mises**; muestra el **factor de
  exageración** (los desplazamientos reales son micras).
- **Oscilación del extremo** $v(L,0,t)$ con marca del instante de impacto.
- **$\sigma_{xx}(x)$** a lo largo de la viga en el instante elegido.
- **Métricas de simulación**: flecha, frecuencia/periodo, $\sigma_{xx}$ y
  $\sigma_{VM}$ máximas con su posición, deformación y **factor de seguridad** con
  veredicto.
- **Validación física**: ver §11.

Mover sliders **no reentrena**: solo reevalúa la red (inferencia paramétrica).

---

## 11. Validación sin *ground truth* (la pregunta del tribunal)

En una PINN **no hay datos de referencia: la verdad son las ecuaciones**. Valido en
cuatro niveles (`utils/metrics.py`, conmutables en `config.VALIDACION`):

1. **Pérdidas finales** desglosadas: ¿cumple residuo, contorno e inicial?
2. **Sanity-checks** físicos: viga plana en $t=0$, empotramiento quieto,
   superficies libres descargadas.
3. **Contraste analítico** con Euler-Bernoulli: la flecha estática debe rondar
   $\delta=PL^3/(2Ec^3)$ y la oscilación, la 1.ª frecuencia natural. (Fórmula
   cerrada, coste nulo; conmutable.)
4. **Residuo en malla nueva**: evaluar el residuo en puntos **no vistos**; si sigue
   bajo, la red **generaliza**; si se dispara, **sobreajustó**. (Autograd 2.º orden,
   por bloques; conmutable.)

Ver [doc 03 §12](03_Respuestas_dudas.md).

---

## 12. Resultados actuales y limitaciones (honestidad de ingeniero)

El **andamiaje completo funciona** (problema, código, app, MLOps, validación), pero
el **modelo entrenado todavía no es físicamente correcto**, y los propios paneles de
validación lo delatan —que es justo para lo que se diseñaron:

- La pérdida se **estanca** (~1.65) y la red converge a una solución **casi trivial**
  ($u,v\approx0$): la flecha PINN ($\sim0.3$ µm) es ~$10^3$ veces menor que la de
  Euler-Bernoulli ($\sim800$ µm).
- La $\sigma_{VM}$ máxima aparece en el **extremo cargado** ($\sim1$ MPa, que es solo
  la tracción impuesta), cuando físicamente debe ser **máxima en el empotramiento**
  ($\sigma_{xx}\approx3PL/2c^2\approx20$ MPa): la **flexión no se ha aprendido**.

**Diagnóstico** (lado deep learning + condicionamiento):

1. **Atractor de la solución trivial.** Tres BC y la IC son homogéneas y empujan a
   cero; la única condición que fuerza solución no nula (la carga) está diluida en
   $w_{\mathrm{bc}}$. El mínimo trivial tiene pérdida $\approx w_{\mathrm{bc}}\cdot
   \overline{\tau_{\mathrm{app}}^2}\approx1.6$ — justo donde se estanca.
2. **Residuo mal condicionado.** El término $U_{\eta\eta}/\hat c^2$ ($\hat c\sim0.1$)
   pesa $\sim100\times$ más que $U_{\hat t\hat t}$; la red prioriza aplanar en $\eta$
   sobre capturar la dinámica.

**Líneas de mejora** (hiperparámetros y formulación, sin tocar el PINNsFormer del
paper): imponer **IC y empotramiento como restricciones duras** (multiplicar la
salida por factores que se anulen en $t=0$ y $x=0$) para eliminar el atractor
trivial; dar un **peso propio y mayor a la BC de carga**; **reescalar el residuo**
para domar el $1/\hat c^2$. Que el sistema de validación **detecte** este fallo de
forma cuantitativa es, en sí, un resultado del proyecto.

---

## 13. Qué demuestra el proyecto

- Dominio del **problema físico** (elastodinámica 2D, tensión plana) y su traducción
  a un **sistema de EDPs**.
- Implementación fiel de una **arquitectura SOTA** (PINNsFormer) y su **extensión no
  trivial** a PINN paramétrica.
- Manejo de las dificultades reales de las PINNs: **adimensionalización**,
  **autograd de 2.º orden**, **memoria**, **balance de pérdidas** y el **atractor
  trivial**.
- Un flujo **MLOps** completo y económico (RunPod + W&B + app) y, sobre todo, un
  **marco de validación sin ground truth** que permite juzgar el modelo con criterio
  científico, no estético.

> En resumen: sé **qué** resuelvo, **cómo** lo resuelvo, **por qué** cada decisión, y
> —lo más importante— **cómo sé si está bien**, incluyendo reconocer con datos cuándo
> todavía no lo está.
