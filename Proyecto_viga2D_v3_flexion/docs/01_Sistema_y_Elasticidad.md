# 01 — El sistema físico y la teoría de la elasticidad

> Objetivo: que entiendas **exactamente** qué se está resolviendo y **todo** el
> vocabulario de mecánica de sólidos que hay detrás. Empezamos por la intuición
> física y terminamos con las ecuaciones que aparecen literalmente en el código.

---

## 1. ¿Qué se resuelve? ¿Es una ODE o una PDE?

Es un sistema de **EDPs (Ecuaciones en Derivadas Parciales)**, no una ODE.

**La razón es directa:** la incógnita es un **campo**, no una función de una sola
variable. Queremos conocer el desplazamiento de **cada punto** $(x, y)$ de la
viga en **cada instante** $t$. Es decir, buscamos dos funciones:

- $u = u(x, y, t)$ → desplazamiento horizontal de cada punto
- $v = v(x, y, t)$ → desplazamiento vertical de cada punto

Cuando una incógnita depende de **varias variables independientes** (aquí tres:
$x$, $y$, $t$) y la ecuación que la gobierna mezcla **derivadas respecto a
varias de ellas** ($\partial^2u/\partial x^2$, $\partial^2u/\partial y^2$,
$\partial^2u/\partial t^2$…), por definición es una **EDP**. Una ODE solo
tendría una variable independiente (p. ej. $u(t)$ con $du/dt$), como en un
péndulo o un circuito RC.

- Es **dinámica** (depende del tiempo) → aparecen derivadas temporales
  $\partial^2/\partial t^2$ (aceleración). Si el problema fuese estático, el
  tiempo desaparecería y sería una EDP elíptica de equilibrio.
- Es **vectorial / acoplada**: hay **dos** ecuaciones (una para $u$, otra para
  $v$) y están acopladas — $u$ aparece en la ecuación de $v$ y viceversa, porque
  estirar el material en una dirección lo contrae en la otra (efecto Poisson).
- Es **2D**: modelamos la viga como una lámina en el plano $x$–$y$ (sección
  longitudinal), no como una línea. Esto permite capturar la distribución de
  tensiones *a través del peralto*, algo que la teoría clásica de vigas
  (Euler-Bernoulli, una ODE) solo aproxima.

> **Frase para la defensa:** "Resuelvo el sistema de EDPs de la elastodinámica
> lineal en 2D (Navier-Cauchy en tensión plana). Es EDP porque la incógnita es
> un campo de desplazamientos $u(x,y,t)$, $v(x,y,t)$, y la ecuación contiene
> derivadas segundas tanto espaciales como temporales."

---

## 2. La cadena conceptual de la elasticidad

Para entender las ecuaciones hay que recorrer una cadena lógica. Cada eslabón
se construye sobre el anterior:

```
Desplazamiento  →  Deformación  →  Tensión  →  Equilibrio (2ª ley de Newton)
   (u, v)            (ε)             (σ)         = ecuación de movimiento
   geometría      derivar         ley de Hooke   derivar + ρ·aceleración
```

Vamos eslabón por eslabón.

### 2.1 Campo de desplazamientos $(u, v)$

Es la incógnita primaria. Para cada partícula material que en reposo está en la
posición $(x, y)$, el vector $(u, v)$ dice **cuánto y hacia dónde se ha movido**.
La posición deformada es $(x + u,\; y + v)$.

- $u$ = movimiento horizontal (a lo largo de la viga, eje axial).
- $v$ = movimiento vertical (la "flecha" o deflexión, lo que típicamente vemos
  pandear).
- Es un **campo**: distinto en cada punto y en cada instante. En el código, la
  red neuronal devuelve precisamente $(U, V)$ — las versiones adimensionales de
  $u, v$ (ver [02](02_PINN_y_DeepLearning.md)).

### 2.2 Deformaciones $\varepsilon$ (strain) — *derivadas del desplazamiento*

El desplazamiento absoluto no genera tensiones (mover toda la viga 1 m no la
rompe). Lo que genera tensión es la deformación **relativa**: cuánto se estira o
se cizalla un punto **respecto a sus vecinos**. Eso son las derivadas espaciales
del desplazamiento. En 2D (tensor de deformación infinitesimal):

$$\varepsilon_{xx} = \frac{\partial u}{\partial x}, \qquad
  \varepsilon_{yy} = \frac{\partial v}{\partial y}, \qquad
  \varepsilon_{xy} = \frac{1}{2}\!\left(\frac{\partial u}{\partial y}+\frac{\partial v}{\partial x}\right)$$

- $\varepsilon_{xx}$: alargamiento unitario en $x$ (estiramiento axial).
- $\varepsilon_{yy}$: alargamiento unitario en $y$ (a través del peralto).
- $\varepsilon_{xy}$: deformación angular / cizalla (distorsión del ángulo recto).

Intuición:
- $\varepsilon_{xx} > 0$: las fibras se alargan en $x$ (tracción); $< 0$: se acortan (compresión).
- $\varepsilon_{xy}$: mide cuánto se "tuerce" un cuadradito del material, perdiendo su
  forma de rectángulo. Es la deformación responsable del **cortante**.

Son **adimensionales** (m/m). Hipótesis de **pequeñas deformaciones**: usamos la
versión lineal del tensor, válida porque los desplazamientos son diminutos
(micras). Esto es lo que hace el problema **lineal** y manejable.

### 2.3 Tensiones $\sigma$ (stress) — *ley de Hooke*

La tensión es **fuerza interna por unidad de área** [Pa = N/m²]. Es la respuesta
del material a la deformación: el material "tira de vuelta" cuando lo deformas,
como un muelle. La **ley de Hooke generalizada** relaciona linealmente tensión y
deformación. En **tensión plana** (ver §4):

$$\sigma_{xx} = (\lambda+2\mu)\,\varepsilon_{xx} + \lambda\,\varepsilon_{yy}$$

$$\sigma_{yy} = \lambda\,\varepsilon_{xx} + (\lambda+2\mu)\,\varepsilon_{yy}$$

$$\tau_{xy} = \mu\!\left(\frac{\partial u}{\partial y}+\frac{\partial v}{\partial x}\right) = 2\mu\,\varepsilon_{xy}$$

En el código, idénticas pero adimensionales, en
[`_stresses`](../physics/pde_loss.py#L82-L91):

$$\hat\sigma_{xx} = (\hat\lambda+2\hat\mu)\,u_x + \hat\lambda\,v_y, \qquad
  \hat\sigma_{yy} = \hat\lambda\,u_x + (\hat\lambda+2\hat\mu)\,v_y, \qquad
  \hat\tau_{xy} = \hat\mu\,(u_y + v_x)$$

Tipos de tensión y su significado físico:
- **$\sigma_{xx}$ (normal axial):** tracción/compresión a lo largo de la viga. En una
  viga flexionada, la cara superior se comprime y la inferior se tracciona (o
  viceversa) → $\sigma_{xx}$ varía linealmente con $y$. **Es la tensión de flexión.**
- **$\sigma_{yy}$ (normal transversal):** apretar/separar en vertical. Suele ser pequeña.
- **$\tau_{xy}$ (cortante):** tensión que "desliza" unas capas sobre otras. Máxima en la
  fibra neutra (centro) y nula en las superficies libres.

### 2.4 Esfuerzos cortantes y flectores (cortante y momento)

Atención al doble vocabulario — esto suele confundir y es buena pregunta de
tribunal:

- **Tensión** ($\sigma$, $\tau$): magnitud **local**, por unidad de área, en *un punto*.
  Es lo que maneja el modelo 2D (campo continuo de tensiones).
- **Esfuerzo cortante $Q(x)$ y momento flector $M(x)$**: magnitudes **integradas**
  de la teoría clásica de vigas (1D). Se obtienen *sumando* las tensiones a lo
  largo del peralto de una sección:

$$Q(x) = \int_A \tau_{xy}\,dA \qquad \text{(resultante de cortante en la sección } x\text{)}$$

$$M(x) = \int_A \sigma_{xx}\cdot y\,dA \qquad \text{(momento flector en la sección } x\text{)}$$

- El **momento flector $M$** mide la tendencia a *curvar* la viga; produce las
  tensiones normales $\sigma_{xx}$ (tracción en una cara, compresión en la otra).
- El **esfuerzo cortante $Q$** mide la tendencia a *cizallar* una sección; produce
  las tensiones tangenciales $\tau_{xy}$.

> **Clave conceptual:** al resolver el campo 2D completo obtenemos las tensiones
> $\sigma_{xx}$, $\tau_{xy}$ punto a punto, que son *más información* que los esfuerzos
> integrados — una **ventaja del enfoque 2D** frente a la teoría de vigas clásica.
> **Sin embargo, $M$ y $Q$ reaparecen como herramienta clave**: el proyecto los usa como
> **restricción de equilibrio seccional** en la pérdida ($\int\tau_{xy}\,dy=Q$,
> $\int\sigma_{xx}\,y\,dy=M$ deben valer la carga y la carga×brazo en toda sección). Es
> lo que rompe el colapso de la PINN a la solución trivial (ver [doc 04 §12](04_Defensa_del_proyecto.md)
> y [doc 05](05_Diagnostico_resultados.md)).

### 2.5 Tensión de Von Mises (lo que pinta la app)

Es un **escalar** que combina todas las componentes del tensor en un único valor
"equivalente", usado para predecir si el material plastifica (criterio de fallo
dúctil). En 2D:

$$\sigma_{VM} = \sqrt{\sigma_{xx}^2 - \sigma_{xx}\,\sigma_{yy} + \sigma_{yy}^2 + 3\,\tau_{xy}^2}$$

En el código: [`inference.py`](../utils/inference.py#L122). El dashboard colorea
la malla deformada por $\sigma_{VM}$ para visualizar de un vistazo dónde se concentra el
esfuerzo (típicamente, **en el empotramiento**).

---

## 3. Los parámetros del material: $E$, $\nu$, $\rho$, $\lambda$, $\mu$

| Símbolo | Nombre | Qué representa | Valor (acero) |
|---|---|---|---|
| $E$ | Módulo de Young | Rigidez axial: cuánta tensión hace falta para una deformación dada. Alto = rígido. | 210 GPa |
| $\nu$ | Coef. de Poisson | Acoplamiento transversal: cuánto se contrae lateralmente al estirar. Adimensional, 0–0.5. | 0.30 |
| $\rho$ | Densidad | Masa por volumen. Aparece en el término inercial $\rho\cdot\text{aceleración}$; gobierna la velocidad de las ondas y la frecuencia de vibración. | 7850 kg/m³ |

De estos se derivan los **parámetros de Lamé**, que son los que aparecen en las
ecuaciones (más cómodos algebraicamente). En **tensión plana**:

$$\mu = \frac{E}{2(1+\nu)} \qquad \text{(segundo parámetro de Lamé = módulo de cortante } G\text{; mide la resistencia a la cizalla)}$$

$$\lambda = \frac{E\nu}{1-\nu^2} \qquad \text{(primer parámetro de Lamé en tensión plana; acopla las deformaciones normales entre sí)}$$

Código: [`config.py`](../config.py#L74-L90). **$\mu$ tiene sentido físico directo**
(módulo de cortante $G$); **$\lambda$** es más algebraico (cuantifica el efecto Poisson
en las ecuaciones).

> **Por qué se entrena solo con acero:** las ecuaciones *adimensionales* dependen
> únicamente de $\nu$ (vía $\hat\mu$, $\hat\lambda$), no de $E$ ni $\rho$. Como $\nu$ del acero (0.30)
> y del aluminio (0.33) son casi iguales, la solución adimensional es
> prácticamente la misma; el material físico se recupera reescalando con $U_{\mathrm{ref}}$,
> $T_{\mathrm{ref}}$. Ver [02 §4](02_PINN_y_DeepLearning.md#4-adimensionalización-por-qué-y-cómo).

---

## 4. Tensión plana vs deformación plana (la hipótesis 2D)

Una viga real es 3D. Reducirla a 2D requiere una hipótesis sobre la dirección
$z$ (el espesor). Hay dos opciones clásicas:

- **Tensión plana (plane stress)** ← **la que usa este TFM.** Supone que el
  cuerpo es **delgado** en $z$ (una placa/lámina fina) y que las caras $z$ están
  libres, de modo que $\sigma_{zz} = \sigma_{xz} = \sigma_{yz} = 0$. Es la hipótesis correcta para una
  viga de poco espesor. Por eso $\lambda = E\nu/(1-\nu^2)$.
- **Deformación plana (plane strain):** para cuerpos muy *largos* en $z$ (una
  presa, un túnel), donde $\varepsilon_{zz} = 0$. Daría $\lambda$ distinta.

Saber distinguirlas y justificar la elección ("viga delgada → tensión plana") es
un punto fino que demuestra dominio.

---

## 5. El dominio, la geometría y las condiciones de contorno

### 5.1 Geometría

```
        y = +c  ┌─────────────────────────────┐  ← superficie libre (arriba)
                │                              │
 empotrada →    │          VIGA 2D             │   ← extremo libre cargado
 (x=0, pared)   │                              │
        y = −c  └─────────────────────────────┘  ← superficie libre (abajo)
                x = 0                        x = L
```

- $x \in [0, L]$: eje axial. $L$ = longitud (1–2 m en el código).
- $y \in [-c, +c]$: eje transversal. $2c$ = **peralto** (canto/altura) de la viga;
  $c$ = semi-altura (5–15 cm). En el código se normaliza a $\eta \in [-1, 1]$.
- $t \in [0, T]$: tiempo.

### 5.2 Condiciones de contorno (CC) e iniciales (CI)

Una EDP no tiene solución única sin condiciones en la frontera del dominio. Aquí
hay cinco, que aparecen en [`pde_loss.py`](../physics/pde_loss.py#L127-L172):

| Frontera | Condición | Significado físico |
|---|---|---|
| **Inicial** $t=0$ | $u=v=0$ y $u_t=v_t=0$ | **Reposo absoluto**: la viga está quieta y sin velocidad antes del impacto. (CI de 2 condiciones porque la EDP es de 2.º orden en $t$.) |
| **Empotramiento** $x=0$ | $u=v=0$ (Dirichlet) | La viga está **soldada/empotrada** a una pared rígida: ese extremo no se mueve. |
| **Superficies** $y=\pm c$ | $\sigma_{yy}=0$, $\tau_{xy}=0$ (Neumann) | **Superficies libres**: nada empuja arriba ni abajo, no hay tensión sobre esas caras. |
| **Extremo libre** $x=L$ | $\sigma_{xx}=0$, $\tau_{xy}=\tau_{\mathrm{app}}(y,t)$ | Extremo donde **se aplica la carga**: sin tracción axial, pero con la tracción cortante de impacto. |

- **Dirichlet** = se fija el *valor* de la incógnita (desplazamiento).
- **Neumann** = se fija la *derivada* / tensión (fuerza).

---

## 6. La carga de impacto: cómo, dónde y qué efecto tiene

Esta es la parte "dinámica" que da vida al problema. Está en
[`applied_traction`](../physics/pde_loss.py#L154-L162).

### 6.1 Dónde se aplica
En el **extremo libre** $x = L$, como una **tracción cortante** (tangencial,
en dirección $y$): un "tirón" transversal que hace vibrar la viga como un
trampolín golpeado en la punta.

### 6.2 Cómo se modela

$$\hat\tau_{\mathrm{app}}(\eta, \hat t) = -\hat\tau_{\mathrm{peak}}
  \cdot \underbrace{(1-\eta^2)}_{\text{perfil parabólico en }y}
  \cdot \underbrace{\min\!\left(\hat t / \hat t_{\mathrm{ramp}},\,1\right)}_{\text{rampa lineal}},
  \qquad \hat\tau_{\mathrm{peak}}=\frac{3P_{\max}/4c}{\sigma_{\mathrm{scale}}}\approx0.015$$

Tres factores multiplicados:

1. **Amplitud $\hat\tau_{\mathrm{peak}}$** (pico adimensional, $\propto P_{\max}$): cuán
   fuerte es el golpe. Es **pequeño** ($\approx0.015$) en el régimen de flexión —la
   tracción motora es mucho menor que la tensión de flexión que induce— lo cual es
   físicamente correcto y, a la vez, la raíz del reto de optimización ([doc 04 §12](04_Defensa_del_proyecto.md)).
2. **Perfil parabólico en $y$: $(1 - \eta^2)$**. La carga **no** es uniforme a través
   del peralto: es máxima en el centro ($\eta=0$) y **nula en las esquinas
   $\eta=\pm1$**. ¿Por qué? Para ser **compatible con las superficies libres**
   ($\tau_{xy}=0$ en $y=\pm c$). Una carga uniforme crearía una discontinuidad
   (singularidad) en las esquinas que arruinaría la convergencia. Esta parábola
   es además el perfil físicamente correcto del cortante en una sección
   rectangular.
3. **Rampa temporal lineal $g(\hat t)=\min(\hat t/\hat t_{\mathrm{ramp}},\,1)$**. La carga
   sube **linealmente** de 0 a $P_{\max}$ en el intervalo $\hat t\in[0,\hat t_{\mathrm{ramp}}]$
   y luego se **mantiene**. Arranca **exactamente en 0** (coherente con el reposo
   inicial), lo que evita excitar un transitorio espurio. Sustituye a la antigua
   rampa sigmoidea: es más fácil de interpretar y, al usarse **idéntica** en la
   PINN y en el FEM de referencia, hace la comparación justa. El único matiz es un
   "codo" (derivada discontinua) al llegar a la meseta, físicamente normal en un
   impacto. **No cambia la EDP**: el perfil temporal es sólo un factor de la BC.

### 6.3 Qué efecto tiene
En $\hat t=0$ la viga está en reposo y la carga es nula. Conforme la rampa sube,
el extremo libre recibe el tirón cortante y la viga empieza a **flexionar y
vibrar**: una onda
elástica viaja desde la punta hacia el empotramiento y se refleja, produciendo
una **oscilación transitoria** (el extremo sube y baja amortiguándose hacia un
equilibrio cuasi-estático). El dashboard muestra esto en la gráfica de
$v(L, 0, t)$ (deflexión de la punta en el tiempo). La **máxima tensión de Von
Mises aparece en el empotramiento**, que es donde el momento flector es mayor.

---

## 7. Las ecuaciones de gobierno (Navier-Cauchy dinámicas)

Reuniendo todos los eslabones (deformación→Hooke→equilibrio + inercia), la 2ª
ley de Newton aplicada a un elemento diferencial da las **ecuaciones de
Navier-Cauchy** (forma en desplazamientos). En unidades físicas
([README](../README.md)):

$$\mu\,(u_{xx}+u_{yy}) + (\lambda+\mu)\,\frac{\partial}{\partial x}(u_x+v_y) = \rho\,u_{tt}$$

$$\mu\,(v_{xx}+v_{yy}) + (\lambda+\mu)\,\frac{\partial}{\partial y}(u_x+v_y) = \rho\,v_{tt}$$

Lectura física término a término:
- $\mu(u_{xx}+u_{yy})$ y el término con $(\lambda+\mu)$: **fuerzas elásticas internas**
  (divergencia del tensor de tensiones) — el material resistiendo la deformación.
- $\rho\,u_{tt}$: **masa × aceleración** — la inercia. Si fuese 0 (estático),
  recuperaríamos el problema de equilibrio.

Esto es literalmente **fuerza elástica = masa × aceleración** para cada punto
del continuo. En el código se resuelve la versión **adimensional** de estas dos
ecuaciones — el "residuo" que la red minimiza
([`loss_residual`](../physics/pde_loss.py#L96-L125)). La forma adimensional, por
qué aparecen las derivadas segundas y cómo la red las calcula se explica en el
**[documento 02](02_PINN_y_DeepLearning.md)**.

---

### ✅ Checklist de dominio (deberías poder explicar sin mirar)
- [ ] Por qué es EDP y no ODE (campo + varias variables independientes).
- [ ] La cadena desplazamiento→deformación→tensión→equilibrio.
- [ ] Diferencia entre tensión local ($\sigma$, $\tau$) y esfuerzos integrados ($M$, $Q$).
- [ ] Qué son $E$, $\nu$, $\rho$, $\lambda$, $\mu$ y por qué solo $\nu$ importa en adimensional.
- [ ] Tensión plana vs deformación plana y por qué eliges la primera.
- [ ] Las 5 condiciones de contorno/iniciales y su tipo (Dirichlet/Neumann).
- [ ] Los 3 factores de la carga de impacto y por qué cada uno es como es.
