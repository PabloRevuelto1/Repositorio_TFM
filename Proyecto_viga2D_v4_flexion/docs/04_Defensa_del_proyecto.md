# 04 — Defensa del proyecto (explicación integral)

> Documento pensado como **guion de defensa** ante tribunal experto en deep
> learning y matemáticas. Explica el proyecto entero —problema, matemáticas,
> método, arquitectura, infraestructura y resultados— con rigor pero a nivel de
> estudiante de máster: completo y honesto, sin academicismo innecesario. Para el
> detalle fino, se enlaza a los docs [01](01_Sistema_y_Elasticidad.md),
> [02](02_PINN_y_DeepLearning.md) y [03](03_Respuestas_dudas.md).

---

> **Actualización (versión simplificada).** Por feedback de los tutores
> (*"impresionante, pero demasiado complejo"*) el proyecto se ha **simplificado**:
> (1) se resuelve **un único escenario fijo** —geometría, carga y material
> constantes— de modo que la red pasa de 7 a **3 entradas** ($\xi,\eta,\hat t$);
> (2) se añade un **ground truth por Elementos Finitos** (`scikit-fem`) con el que
> la app **compara la PINN objetivamente**; (3) la carga es ahora una **rampa
> lineal** en vez de una sigmoide. La discusión de la versión *paramétrica*
> original se conserva como justificación del compromiso de diseño.

## 1. En una frase

Resuelvo la **vibración de una viga empotrada 2D golpeada por un impacto** con una
**red neuronal informada por la física** (PINN), usando la arquitectura
**PINNsFormer** (un Transformer, Zhao et al., ICLR 2024), y la **valido contra una
solución de referencia por Elementos Finitos**, todo servido en un **dashboard
interactivo** con flujo MLOps (entreno en GPU cloud, modelo en W&B, visualización
y comparación local).

---

## 2. El problema físico

Una viga de longitud $L$ y semi-altura $c$ está **empotrada** en $x=0$ (pared) y
**libre** en $x=L$. En el extremo libre recibe una **carga cortante de impacto**:
un tirón transversal que crece como una **rampa lineal** de 0 a $P_{\max}$ y luego
se mantiene. Queremos saber **cómo se desplaza y se tensiona la viga en el
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
$\tau_{xy}=\tau_{\mathrm{app}}$ en $x=L$), con perfil parabólico $(1-\eta^2)$ y **rampa
lineal** $g(\hat t)=\min(\hat t/\hat t_{\mathrm{ramp}},1)$ (sube de 0 a $P_{\max}$ y se
mantiene; arranca en 0, coherente con el reposo).

---

## 4. Por qué una PINN (y qué papel juega el FEM)

Una **PINN** invierte la lógica del deep learning supervisado: la red **es** la
solución $\mathcal N_\theta(x,y,t)\to(u,v)$, y la **pérdida es el residuo de la
física**, no un error contra etiquetas. No necesita datos de referencia para
entrenar: se minimiza cuánto incumplen las ecuaciones la red y sus condiciones.

$$\mathcal L = w_{\mathrm{res}}\underbrace{\lVert r_u\rVert^2+\lVert r_v\rVert^2}_{\text{residuo EDP}}
   + w_{\mathrm{load}}\,\mathcal L_{\mathrm{load}} + w_{\mathrm{bc}}\,\mathcal L_{\mathrm{bc}} + w_{\mathrm{ic}}\,\mathcal L_{\mathrm{ic}}$$

donde $r_u=\rho\,\partial_{tt}u-(\dots)$ es el residuo: cuánto se aparta la red de
cumplir Navier-Cauchy en cada punto. **Ventajas frente a FEM:** malla-libre,
solución continua y derivable (derivadas exactas por autograd) y consulta puntual
instantánea. El precio: el entrenamiento es un problema de optimización no convexo,
delicado (ver §11).

**El FEM no es el rival, es el juez.** En este proyecto resolvemos *además* el
mismo problema por **Elementos Finitos** (`fem/fem_solver.py`, `scikit-fem` con
integración temporal de **Newmark**) como **solución de referencia objetiva**.
No entrena la PINN: sirve para **medir cuánto acierta**. Así combinamos lo mejor
de cada mundo: la PINN como solucionador aprendido y el FEM como verdad física
contra la que validar (la app declara un "ganador por categoría"). Es la respuesta
directa a *"¿cuál es tu ground truth?"*.

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
$\sigma_{\mathrm{scale}}=P_{\mathrm{ref}}L_{\mathrm{ref}}^2/(2c_{\mathrm{ref}}^3)$
(**tensión de flexión** $M c/I$) y $U_{\mathrm{ref}}=L_{\mathrm{ref}}\sigma_{\mathrm{scale}}/E$.
Así los coeficientes de la EDP quedan de orden unidad ($\hat\mu\approx0.385$,
$\hat\lambda\approx0.33$) **y el desplazamiento de flexión también es O(1)**.

> **Dos escalas que hay que acertar (lección aprendida, [doc 05](05_Diagnostico_resultados.md)).**
> (1) *Tensión:* adimensionalizar por la **flexión** ($Mc/I$), no por la tracción
> aplicada ($3P/4c$); si no, la flecha adimensional vale $\sim(L/c)^2\!\approx\!225$ y la
> red no la alcanza. (2) *Tiempo:* $T_{\mathrm{ref}}$ es el tiempo de **onda**
> ($\sim0.3$ ms), $\sim50\times$ más rápido que la **flexión** ($\sim13.5$ ms); el
> horizonte $\hat T=100$ cubre $>1$ periodo de flexión (antes $\hat T=4=0.77$ ms sólo
> capturaba ondas, sin flexión). Acertar ambas es lo que hace que **PINN, FEM y
> Euler-Bernoulli coincidan**.

El selector de material de la app reescala la solución adimensional a unidades físicas
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

**Adaptación a este problema (escenario fijo):** la entrada es de dimensión **3**:
$(\xi,\eta,\hat t)$ —sólo coordenadas, porque la geometría/carga es constante—.
Salida: $(U,V)$. Además, la salida se multiplica por el **ansatz de restricción
dura** $\xi\,\dfrac{\hat t^2}{\hat t^2+\tau^2}$, que anula exactamente el desplazamiento
en el empotramiento ($\xi=0$) y en el instante inicial ($\hat t=0$, con velocidad nula,
porque cerca de 0 va como $(\hat t/\tau)^2$), imponiendo esas condiciones por
construcción. La envolvente temporal **satura a 1** para $\hat t\gg\tau$: a diferencia
del antiguo $(\hat t/\hat T)^2$ —que sobre el horizonte largo de flexión crecía
$\times\hat T^2$ y descondicionaba el problema— ésta no crece. Regla general: la
envolvente de un *hard-constraint* nunca debe crecer sin cota sobre el dominio.
*(La versión paramétrica original usaba $d_{in}=7$ con los 4 parámetros muestreados
por Latin Hypercube; ver [doc 02 §6](02_PINN_y_DeepLearning.md).)*

---

## 7. La función de pérdida en detalle

`physics/pde_loss.py` calcula **cinco** bloques sobre conjuntos de puntos distintos:

- $\mathcal L_{\mathrm{res}}$: residuo de Navier-Cauchy en el **interior** (usa
  derivadas 2.ª; **reescalado por $\hat c$** para equilibrar su magnitud con las demás
  pérdidas — antes $\hat c^2$, que lo hundía a $\sim10^{-4}$ e infra-imponía la EDP).
- $\mathcal L_{\mathrm{load}}$: **extremo cargado** (condición motora, peso propio).
- $\mathcal L_{\mathrm{bc}}$: empotramiento + superficies libres (homogéneas).
- $\mathcal L_{\mathrm{ic}}$: reposo en $t=0$ (redundante con la restricción dura;
  queda como monitor ≈0).
- $\mathcal L_{\mathrm{eq}}$: **equilibrio seccional** (cortante/momento integral por
  sección). Es **lo que de verdad rompe el colapso trivial**; ver §12.

Pesos actuales: $w_{\mathrm{res}}=1,\ w_{\mathrm{load}}=20,\ w_{\mathrm{bc}}=5,\
w_{\mathrm{ic}}=10,\ w_{\mathrm{eq}}=3$ (+ $\mathcal L_{\mathrm{eq}}$ con relajación
temporal). El balance es **crítico**: la carga (peso alto) y el equilibrio seccional evitan
el colapso a $u=v=0$; $w_{\mathrm{ic}}$ es alto porque ahora impone la **velocidad inicial**
$u̇(0)=0$ (antes la fijaba el ansatz), motor de la oscilación (ver §12).

---

## 8. Entrenamiento

- **Muestreo**: Latin Hypercube (mejor cobertura que aleatorio puro) del dominio
  espacio-temporal $(\xi,\eta,\hat t)$; 6000 puntos de residuo, 1000/frontera, 1500 de
  IC, y 400 **secciones** ($\times17$ nodos en $\eta$) para el equilibrio seccional.
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

- **Escenario fijo**: tabla con $L,c,P_{\max},\hat t_{\mathrm{ramp}},\hat T$ y
  material (constantes), y botón **Simular y comparar**.
- **Métricas de entrenamiento**: semáforo de calidad, pérdidas finales
  ($\mathcal L_{\mathrm{res}},\mathcal L_{\mathrm{load}},\mathcal L_{\mathrm{bc}},
  \mathcal L_{\mathrm{ic}}$), convergencia, nº de parámetros y épocas.
- **Malla deformada animada** por **Von Mises**, **conmutable PINN/FEM** (misma
  escala de color); muestra el **factor de exageración** (los desplazamientos son µm).
- **Oscilación del extremo** $v(L,0,t)$ y **$\sigma_{xx}(x)$**: curvas **PINN y FEM
  superpuestas**.
- **🏆 PINN vs FEM**: tabla de **ganador por categoría** (tiempo, flecha, $\sigma_{VM}$,
  ubicación de la tensión máxima, fidelidad, consulta malla-free) + **error L2** de
  los campos de la PINN contra el FEM.
- **Métricas de simulación** y **validación física**: ver §11.

Pulsar Simular **no reentrena**: reevalúa la red y reutiliza el FEM cacheado.

---

## 11. Validación: ahora SÍ hay *ground truth* (la pregunta del tribunal)

La pregunta clásica a una PINN es *"¿cuál es tu ground truth?"*. En esta versión la
respuesta es contundente: el **FEM**. Valido en varios niveles complementarios
(`utils/metrics.py`, conmutables en `config.VALIDACION`):

0. **Ground truth FEM** (`fem/fem_solver.py`): el **mismo** problema resuelto por
   Elementos Finitos sobre la **misma malla e instantes**. La app mide el **error
   L2** de los campos de la PINN contra el FEM y enfrenta ambos métodos con un
   **ganador por categoría** (precisión, tiempo de cómputo, ubicación de la tensión
   máxima…). Es la validación de referencia.
1. **Pérdidas finales** desglosadas ($\mathcal L_{\mathrm{res}},\mathcal L_{\mathrm{load}},
   \mathcal L_{\mathrm{bc}},\mathcal L_{\mathrm{ic}},\mathcal L_{\mathrm{eq}}$). **Aviso:
   la pérdida TOTAL no es el termómetro** (un mínimo trivial degenerado puede dar pérdida
   bajísima con un resultado pésimo, ver §12.2); el veredicto real es el **error L2 vs
   FEM** del nivel 0.
2. **Sanity-checks** físicos: viga plana en $t=0$, empotramiento quieto,
   superficies libres descargadas.
3. **Contraste analítico** con Euler-Bernoulli: la flecha estática debe rondar
   $\delta=PL^3/(3EI)$ y la oscilación, la 1.ª frecuencia natural. (Fórmula cerrada;
   complementaria, conmutable.)
4. **Residuo en malla nueva**: evaluar el residuo en puntos **no vistos**; si sigue
   bajo, la red generaliza. (Autograd 2.º orden, por bloques; conmutable.)

Ver [doc 03 §12](03_Respuestas_dudas.md).

---

## 12. Resultados, diagnóstico y solución (la parte fuerte de la defensa)

El **andamiaje completo funciona** (problema, código, app, MLOps, validación). La
historia de los **resultados** es, además, el mejor material de defensa: es un caso real
de *debugging* físico-numérico documentado de principio a fin
([doc 05](05_Diagnostico_resultados.md)). Tres actos.

### 12.1 Acto I — Régimen equivocado (ondas en vez de flexión)

Los primeros entrenamientos daban resultados absurdos y, sobre todo, **incoherentes
entre sí**: el FEM daba 68 µm de flecha, Euler-Bernoulli 803 µm (¡$\times12$!). La pista
la resolvieron las **escalas de tiempo**: $T_{\mathrm{ref}}=L\sqrt{\rho/E}$ es el tiempo
de **onda elástica** ($\sim0.3$ ms), no el de **flexión** ($\sim13.5$ ms). Con $\hat T=4$
el horizonte era 0.77 ms = **5.7 % de un periodo de flexión**: la viga ni flexaba, sólo
se veían **ondas**. Validar eso contra Euler-Bernoulli (teoría de flexión) era una
comparación inválida (de ahí los "errores" del 99 %). **Solución:** llevar al **régimen
de flexión** ($\hat T=100$) y reescalar la tensión a la de flexión ($Mc/I$). Con esto el
**FEM quedó perfecto**: flecha $\sim1400$ µm oscilando, $\sigma_{VM}$ máx. $\approx48$ MPa
**en el empotramiento** (como debe ser). PINN, FEM y Euler-Bernoulli ya coinciden en
orden de magnitud.

### 12.2 Acto II — Colapso a la solución trivial

Con el régimen correcto, la **PINN** seguía fallando, pero de una forma muy reveladora:
pérdida total **bajísima** ($2\cdot10^{-5}$) y, sin embargo, flecha 15 µm, $\sigma_{VM}$
1.3 MPa **sólo en la punta** (= la tracción impuesta), error L2 vs FEM 99 %. La red
colapsaba a una solución **casi trivial** ($u\approx0$).

**Diagnóstico (modo de fallo intrínseco, no un bug):** el problema es de flexión
**controlada por fuerza** y la EDP del interior es **homogénea**
($\rho\ddot u=\nabla\!\cdot\sigma$, sin fuerza de volumen) → $u\approx0$ la satisface con
residuo $\approx0$. Lo único que debería generar la gran flexión es una BC de Neumann
**débil** (la tracción $\hat\tau\approx0.015$). El descenso por gradiente se queda en la
cuenca trivial, y **la pérdida total no lo detecta** (mínimo degenerado): por eso el
semáforo decía "BUENO" con un resultado pésimo. Es un fallo conocido de las PINN
*vanilla* en elasticidad accionada por fuerza (la FEM no lo sufre porque su matriz de
rigidez global $K$ acopla todo el dominio; la PINN sólo tiene residuo local).

### 12.3 Acto III — Equilibrio seccional (la solución, física pura)

La cura es **imponer el equilibrio integral que el residuo local no transmite**. Para una
ménsula con carga en la punta, en **toda** sección $\xi$ se cumple (consecuencia exacta
de $\nabla\!\cdot\sigma=0$, **no son datos**):

$$\underbrace{\int_{-1}^{1}\hat\tau_{xy}\,d\eta = -K_{\mathrm{shear}}\,g(\hat t)}_{\text{cortante resultante}=\,\text{carga aplicada}}, \qquad \underbrace{\int_{-1}^{1}\hat\sigma_{xx}\,\eta\,d\eta = +K_{\mathrm{moment}}(1-\xi)\,g(\hat t)}_{\text{momento flector}=\,\text{carga}\,\times\,\text{brazo}}$$

con $K_{\mathrm{shear}}=P/(\Sigma c)=0.02$ y $K_{\mathrm{moment}}=PL/(\Sigma c^2)=0.3$
(signo y forma **verificados contra el FEM**). Estas son exactamente el **esfuerzo
cortante $Q(x)$ y el momento flector $M(x)$** de la teoría de vigas ([doc 01 §2.4](01_Sistema_y_Elasticidad.md)),
ahora usados como **restricción**. Con $u\approx0$ el momento integral da $0$ en vez de
$0.3(1-\xi)$ → penalización grande → **la red sale del trivial**. Se impone **blando**
($w_{\mathrm{eq}}=20$): fija el esqueleto cuasi-estático de la flexión; la oscilación
dinámica (inercia) la sigue aportando el residuo. Implementación: `loss_equilibrium`
(cuadratura en $\eta$ por secciones) en `physics/pde_loss.py`.

> **Lección para el tribunal.** No es "la PINN no funciona": es saber **por qué** un
> resultado es malo (régimen + colapso por EDP homogénea), **medirlo** con el termómetro
> correcto (error L2 vs FEM, no la pérdida total) y **resolverlo con física** (equilibrio
> integral derivado de $\nabla\!\cdot\sigma=0$ y validado contra el FEM). Eso es
> ingeniería de verdad.

### 12.4 Acto IV — De la flexión estática a la dinámica (causalidad temporal)

Roto el colapso, la PINN acierta la **flexión estática** (833 µm, Euler-Bernoulli al 13 %)
pero **no oscila**: la punta baja y se queda, mientras el FEM vibra. Razón profunda: la
oscilación es la **vibración libre** de la viga, y un modo libre tiene **residuo de EDP = 0**
→ la pérdida no distingue la solución cuasi-estática (que no rebota) de la dinámica; ambas
son mínimos, y la "simple" gana. El FEM sí oscila porque **marcha en el tiempo** (causalidad
incorporada). Solución: **inyectar causalidad** sin tocar el paper — (a) **CI de velocidad
blanda** (el ansatz pasa a $\xi\,\hat t/(\hat t+\tau)$, que anula sólo el desplazamiento; la
red impone $u̇(0)=0$ por pérdida → excita el modo libre); (b) **relajar el equilibrio
seccional** en $\hat t$ tardío (su objetivo es cuasi-estático); (c) **currículo temporal**
(entrenar con ventana $[0,\hat t]$ creciente, como el marcha-en-tiempo del FEM). *(Estado:
implementado y validado con lotes pequeños; pendiente el reentreno de confirmación.)*

---

## 13. Qué demuestra el proyecto

- Dominio del **problema físico** (elastodinámica 2D, tensión plana) y su traducción
  a un **sistema de EDPs**.
- Implementación fiel de una **arquitectura SOTA** (PINNsFormer) y su adaptación al
  problema (restricciones duras, escenario fijo; con la variante paramétrica documentada).
- Manejo de las dificultades reales de las PINNs: **adimensionalización** (y la elección
  correcta de **escalas** de tiempo y desplazamiento), **autograd de 2.º orden**,
  **memoria**, **balance de pérdidas**, el **régimen físico** (ondas vs flexión) y el
  **colapso trivial** en problemas controlados por fuerza —diagnosticado con física
  (EDP homogénea) y resuelto con un **constraint de equilibrio seccional** verificado
  contra el FEM.
- Un flujo **MLOps** completo y económico (RunPod + W&B + app) y, sobre todo, una
  **validación con ground truth por FEM** (más *sanity-checks* y Euler-Bernoulli)
  que permite juzgar el modelo con criterio científico, no estético.

> En resumen: sé **qué** resuelvo, **cómo** lo resuelvo, **por qué** cada decisión, y
> —lo más importante— **cómo sé si está bien**, incluyendo reconocer con datos cuándo
> todavía no lo está.
