# 03 — Respuestas a dudas concretas

> Resuelve siete preguntas puntuales surgidas al leer el
> [documento 01](01_Sistema_y_Elasticidad.md). Mismo estilo didáctico y profundo.

---

## 1. ¿Qué es un campo de desplazamientos?

Un **campo** (en física/matemáticas) es una **función que asigna un valor a cada
punto del espacio** (y aquí, también del tiempo). Un campo de desplazamientos
asigna a cada punto material de la viga **el vector que indica cuánto se ha
movido** desde su posición de reposo.

Aclaremos tu intuición, porque mezclas dos ideas:

- **No** es "el desplazamiento de un punto a lo largo del tiempo" (eso sería una
  *trayectoria*, función solo de $t$).
- **Sí** es: para **cada** punto $(x,y)$ y **cada** instante $t$, un vector de
  desplazamiento. Es una función de las tres variables a la vez:

$$\vec{d}(x,y,t) = \big(u(x,y,t),\; v(x,y,t)\big)$$

Sobre tu pregunta "¿campo horizontal en un punto o campo general de la viga?":
son **niveles distintos del mismo objeto**, y conviene tener el vocabulario claro:

| Término | Qué es | Ejemplo |
|---|---|---|
| **Componente** del campo | una de las dos funciones, $u$ (horizontal) o $v$ (vertical) | $u(x,y,t)$ |
| **Valor** del campo en un punto-instante | el vector concreto ahí | $\vec d(0.5,\,0,\,2) = (3\,\mu m,\,-50\,\mu m)$ |
| **Campo** (sin más) | **toda** la función, los desplazamientos de **todos** los puntos en **todos** los instantes | $\vec d(x,y,t)$ |

Así que "el campo de desplazamientos de la viga" se refiere al **objeto completo**
(todos los puntos, todos los instantes), mientras que "el campo de
desplazamientos horizontal" se refiere a **una componente** ($u$) de ese objeto.
La red neuronal del proyecto aprende precisamente este campo completo: le das
$(x,y,t)$ y te devuelve $(u,v)$ ahí.

---

## 2. El efecto Poisson

El **efecto Poisson** es un hecho experimental: cuando estiras un material en una
dirección, **se contrae en las direcciones perpendiculares** (y al revés: si lo
comprimes, se ensancha).

> Imagen mental: una goma elástica. Al estirarla longitudinalmente, **se
> adelgaza**. Una plastilina: al aplastarla por arriba, **se ensancha** por los
> lados.

Se cuantifica con el **coeficiente de Poisson** $\nu$ (la `nu` del código,
$\nu=0.30$ para acero):

$$\nu = -\frac{\text{deformación transversal}}{\text{deformación axial}}
     = -\frac{\varepsilon_{\text{transv}}}{\varepsilon_{\text{axial}}}$$

El signo menos hace que $\nu$ sea positivo: estirar ($\varepsilon_{axial}>0$)
produce contracción lateral ($\varepsilon_{transv}<0$). Valores típicos: acero
$0.30$, aluminio $0.33$, caucho $\approx0.5$ (casi incompresible), corcho
$\approx0$ (no se ensancha, por eso entra bien en la botella).

**Por qué esto acopla las ecuaciones:** mira la ley de Hooke del proyecto:

$$\sigma_{xx} = (\lambda+2\mu)\,\varepsilon_{xx} + \lambda\,\varepsilon_{yy}$$

La tensión horizontal $\sigma_{xx}$ **no depende solo** de la deformación
horizontal $\varepsilon_{xx}=\partial u/\partial x$, sino **también** de la
vertical $\varepsilon_{yy}=\partial v/\partial y$, vía $\lambda$ (que contiene
$\nu$). Es decir, lo que pasa con $v$ influye en la ecuación de $u$ y viceversa:
**están acopladas**. El término $\lambda$ es, físicamente, el mensajero del
efecto Poisson dentro de las ecuaciones. Si $\nu=0$ (sin efecto Poisson),
$\lambda$ no se anula del todo pero el acoplamiento se reduce drásticamente.

---

## 3. La teoría clásica de vigas (Euler-Bernoulli)

Es el modelo **simplificado** de vigas que se enseña en resistencia de materiales,
anterior al enfoque 2D/3D del continuo. En lugar de tratar la viga como una
lámina con campo $(u,v)$ en cada punto, la reduce a **una línea** (el eje de la
viga) y describe su flexión con **una sola función**: la flecha $w(x)$ —cuánto
baja el eje en cada posición $x$.

Su ecuación de gobierno es una **ODE** (una variable independiente, $x$):

$$EI\,\frac{d^4 w}{dx^4} = q(x)$$

donde $E$ es el módulo de Young, $I$ el momento de inercia de la sección y $q$ la
carga distribuida. Es **mucho más barata** de resolver que el sistema 2D.

**Por qué "solo aproxima":** Euler-Bernoulli asume hipótesis fuertes:

- Las **secciones planas permanecen planas y perpendiculares** al eje tras
  deformarse → **desprecia la deformación por cortante** ($\tau_{xy}$). Buena
  para vigas **esbeltas** (largas y finas), mala para vigas **cortas y altas**,
  donde el cortante importa.
- No captura cómo se **distribuyen las tensiones a través del peralto** con
  detalle (las da por una fórmula lineal supuesta, no resuelta).

El modelo 2D de este TFM **no hace esas suposiciones**: resuelve las ecuaciones
del continuo completas (Navier-Cauchy), así que captura el cortante, las
concentraciones de tensión en el empotramiento, etc. Por eso decimos que
Euler-Bernoulli "solo aproxima" lo que aquí se resuelve de forma más fiel.
(Existe un modelo intermedio, **Timoshenko**, que sí incluye cortante pero sigue
siendo 1D.)

---

## 4. Elastodinámica lineal y por qué tensión plana

Desglosemos el nombre completo del problema, término a término:

**"Elasto-"** → **elasticidad**: el material se deforma bajo carga pero
**recupera** su forma original al retirarla (como un muelle), sin daño
permanente. Lo opuesto sería plasticidad (deformación permanente).

**"-dinámica"** → **depende del tiempo** e incluye la **inercia**
($\rho\,\partial^2 u/\partial t^2$, masa × aceleración). Hay ondas y vibraciones.
Lo opuesto sería **estática** (equilibrio, sin tiempo). Como el problema modela un
**impacto** y la **oscilación** posterior, es dinámica.

**"lineal"** → hay **proporcionalidad** en dos sentidos:
1. Tensión ∝ deformación (ley de Hooke, material lineal).
2. Deformación ∝ derivadas del desplazamiento (tensor de deformación *lineal*,
   válido por pequeñas deformaciones — ver pregunta 5).

Consecuencia: las ecuaciones son lineales en la incógnita, lo que las hace
**tratables** y garantiza buenas propiedades (superposición, unicidad). Lo
opuesto sería no-lineal (grandes deformaciones, material que plastifica…).

**Por qué "tensión plana" (plane stress):**
Una viga real es 3D, pero la modelamos en el plano $x\text{–}y$. Para reducir 3D→2D
hay que suponer algo sobre el espesor (dirección $z$). Hay dos hipótesis clásicas:

- **Tensión plana** ← la de este TFM. Para cuerpos **delgados** en $z$ (una placa
  o viga de poco espesor) con las caras $z$ libres: no hay nada que las empuje, así
  que las tensiones fuera del plano son nulas: $\sigma_{zz}=\sigma_{xz}=\sigma_{yz}=0$.
  El material **sí puede contraerse libremente** en $z$ (Poisson). De aquí sale
  $\lambda = E\nu/(1-\nu^2)$, exactamente la fórmula del
  [código](../config.py#L79-L80).
- **Deformación plana** (plane strain): para cuerpos **muy largos** en $z$ (una
  presa, un túnel, una tubería larga), donde el material **no puede** moverse en
  $z$: $\varepsilon_{zz}=0$. Daría una $\lambda$ distinta.

Se elige **tensión plana** porque una viga es un elemento **delgado** comparado
con su longitud y canto; sus caras laterales están libres. Justificar esto
("viga delgada de caras libres ⇒ tensión plana") es un punto fino de tribunal.

---

## 5. La deformación angular $\varepsilon_{xy}$, el cortante y por qué linealizar

### 5.1 Qué significa $\varepsilon_{xy}=\tfrac12(\partial u/\partial y+\partial v/\partial x)$

Las deformaciones $\varepsilon_{xx}$ y $\varepsilon_{yy}$ miden **cambios de
longitud** (estirar/encoger). Pero un material puede deformarse **sin cambiar
ninguna longitud**: solo **torciendo los ángulos**. Eso es la deformación
angular (o de cizalla) $\varepsilon_{xy}$.

Imagina un **cuadradito** dibujado en el material, con sus lados paralelos a los
ejes y sus esquinas a 90°:

```
   antes (reposo)          después (cizalla pura)
   ┌─────────┐                 ╱─────────╱
   │         │                ╱         ╱
   │         │               ╱         ╱
   └─────────┘              ╱─────────╱
   ángulos = 90°          ángulos ≠ 90°  (se ha "inclinado")
```

El cuadrado se convierte en un **romboide**: sus lados ya no forman 90°.
$\varepsilon_{xy}$ mide **cuánto se ha cerrado/abierto ese ángulo recto**. Los dos
términos explican por qué:

- $\partial u/\partial y$: cuánto cambia el desplazamiento **horizontal** $u$ a
  medida que subes en **vertical** $y$ → inclina los lados verticales.
- $\partial v/\partial x$: cuánto cambia el desplazamiento **vertical** $v$ al
  avanzar en **horizontal** $x$ → inclina los lados horizontales.

La suma de ambas inclinaciones = distorsión total del ángulo. El $\tfrac12$ es una
convención para que el tensor sea simétrico y consistente.

### 5.2 Qué es el cortante y por qué $\varepsilon_{xy}$ es responsable de él

**Cortante** (o cizalla, *shear*) es el tipo de esfuerzo que **desliza unas capas
del material sobre las otras**, en paralelo (tangencial), en lugar de
estirarlas/comprimirlas (perpendicular). Piensa en:

- Una **baraja de cartas**: si empujas la carta de arriba hacia un lado, las
  cartas **deslizan** unas sobre otras → eso es cizalla.
- Unas **tijeras** (de hecho "cortante" viene de cortar): las dos hojas aplican
  fuerzas paralelas pero en sentidos opuestos y desfasadas → el papel se cizalla
  y se separa.

La conexión con $\varepsilon_{xy}$ es directa vía la ley de Hooke del proyecto:

$$\tau_{xy} = 2\mu\,\varepsilon_{xy} = \mu\Big(\frac{\partial u}{\partial y}+\frac{\partial v}{\partial x}\Big)$$

La **tensión cortante** $\tau_{xy}$ (fuerza tangencial por área) es directamente
proporcional a la **deformación angular** $\varepsilon_{xy}$ (la distorsión del
ángulo). Por eso decimos que $\varepsilon_{xy}$ "es la deformación responsable del
cortante": si no hay distorsión angular, no hay tensión cortante. El factor de
proporción es $\mu$ (el módulo de cortante $G$).

### 5.3 Por qué la hipótesis de pequeñas deformaciones / problema lineal

El tensor de deformación **exacto** (válido para grandes deformaciones) tiene
términos **cuadráticos**, p. ej.:

$$\varepsilon_{xx}^{\text{exacto}} = \frac{\partial u}{\partial x}
   + \tfrac12\Big[\big(\tfrac{\partial u}{\partial x}\big)^2
                 +\big(\tfrac{\partial v}{\partial x}\big)^2\Big]$$

Cuando los desplazamientos y sus gradientes son **diminutos** (aquí, micras: los
gradientes son $\sim10^{-5}$), esos términos cuadráticos son **despreciables**
($(10^{-5})^2 = 10^{-10}$, irrelevante frente a $10^{-5}$). Así nos quedamos con
la **versión lineal**: $\varepsilon_{xx}\approx\partial u/\partial x$. Esto es la
**hipótesis de pequeñas deformaciones**.

**Por qué nos interesa que el problema sea lineal:**

1. **Validez física:** los desplazamientos reales de la viga *son* minúsculos,
   así que la linealización no introduce error apreciable; sería absurdo cargar
   con la complejidad no-lineal sin ganancia.
2. **Tratabilidad matemática:** las ecuaciones lineales tienen **solución única**,
   permiten **superposición** (sumar soluciones) y están bien condicionadas.
3. **Entrenamiento de la PINN:** un problema lineal da un **paisaje de pérdida**
   más benigno y derivadas más estables, lo que ayuda a que la red **converja**.
   Términos cuadráticos meterían no-linealidades fuertes que complicarían la
   optimización.

En resumen: pequeño desplazamiento → tensor lineal → problema lineal → físicamente
fiel **y** numéricamente manejable. Ganamos en los dos frentes.

---

## 6. Interpretación física de cada tensión (la pregunta clave)

Aquí está la confusión más importante de resolver, así que vamos despacio y con
imágenes. La clave que desbloquea todo: **en una viga en flexión, la tensión
$\sigma_{xx}$ NO es uniforme — cambia de signo a lo largo del peralto.**

### 6.1 $\sigma_{xx}$: por qué la flexión es tracción Y compresión a la vez

Tu intuición es correcta y a la vez incompleta. Tienes razón en que:
- **Traccionar** de los dos extremos (tirar) → alargamiento axial uniforme.
- **Comprimir** de los dos extremos (empujar) → acortamiento (y posible pandeo).

Eso es **carga axial pura**, y ahí $\sigma_{xx}$ es **igual en todo el peralto**.
Pero la **flexión es otra cosa**. Cuando la viga se **dobla** (como en este TFM,
por una carga transversal en la punta), pasa lo siguiente:

```
        viga doblada hacia abajo (la punta baja)
   ───────────────────────────────────  ← fibra SUPERIOR: se ESTIRA  → tracción (σ_xx > 0)
   ─────────────────────────────────    ← fibra NEUTRA:   ni estira ni encoge (σ_xx = 0)
   ───────────────────────────────      ← fibra INFERIOR: se COMPRIME → compresión (σ_xx < 0)
```

Cuando doblas la viga hacia abajo, las fibras de **arriba** tienen que recorrer un
arco más largo → **se estiran** (tracción), y las de **abajo** un arco más corto →
**se comprimen**. Entre ambas hay una capa, la **fibra neutra**, que no cambia de
longitud. Por eso:

$$\sigma_{xx}(y) \;\text{varía linealmente con } y:\quad
  \text{tracción arriba},\;\; 0 \text{ en el centro},\;\; \text{compresión abajo}.$$

**Así que la flexión genera $\sigma_{xx}$ sin necesidad de empujar/tirar
axialmente:** la causa es el **doblado**, no una fuerza en $x$. La misma viga
tiene, en una sección, tracción y compresión simultáneas en distintas alturas. Por
eso $\sigma_{xx}$ se llama "tensión de flexión" y es la que produce el momento
flector $M=\int \sigma_{xx}\,y\,dA$ (pregunta de doc 01 §2.4).

### 6.2 $\sigma_{yy}$: apretar/separar en vertical

$\sigma_{yy}$ es la tensión normal en dirección **vertical** ($y$): mide si el
material está siendo **apretado o estirado de arriba abajo**, perpendicular al eje
de la viga.

```
   fuerza ↓
   ▼▼▼▼▼▼
   ┌──────┐
   │      │   ← el material se "aplasta" verticalmente: σ_yy de compresión
   └──────┘
   ▲▲▲▲▲▲
   fuerza ↑
```

Surge, por ejemplo, justo **debajo de donde se aplica una carga** que empuje
contra una cara. En una viga esbelta en flexión suele ser **pequeña** comparada
con $\sigma_{xx}$ (por eso doc 01 dice "suele ser pequeña"), pero no es cero cerca
de los puntos de aplicación de carga y apoyos.

### 6.3 $\tau_{xy}$: el deslizamiento de capas (cizalla) y "cizallar una sección"

$\tau_{xy}$ es la **tensión cortante**: la fuerza **tangencial** (paralela a la
sección) que tiende a hacer **deslizar una capa de material sobre la contigua**.
Retoma la imagen de la **baraja de cartas** (pregunta 5.2):

```
   imagina la viga como un montón de láminas horizontales apiladas:

   ════════════  →  fuerza tirando hacia la derecha arriba
   ════════════
   ════════════  ←  fuerza hacia la izquierda abajo
   las láminas tienden a DESLIZAR unas sobre otras → eso resiste τ_xy
```

"**Cizallar una sección**" significa aplicar fuerzas que tiendan a **cortar
transversalmente** la viga, como unas tijeras: empujar el material de un lado de
un plano hacia arriba mientras el otro lado va hacia abajo. La sección sufre un
esfuerzo que la tiende a **rebanar**.

¿Cómo aparece físicamente aquí? Es justo la **carga de impacto** del TFM: se
aplica una **tracción cortante** $\tau_{app}$ en el extremo libre (un tirón
**transversal** en la punta). Esa fuerza tangencial se transmite por la viga como
tensión cortante $\tau_{xy}$, que es **máxima en la fibra neutra** (el centro,
$y=0$) y **nula en las superficies libres** ($y=\pm c$) — justo el perfil
parabólico $(1-\eta^2)$ que impone el código en
[`applied_traction`](../physics/pde_loss.py#L154-L162). Físicamente: en las caras
libres no hay nada que empuje tangencialmente, así que ahí el cortante debe
anularse; en el centro es donde más "deslizan" las capas.

### 6.4 Tabla resumen de las tres tensiones

| Tensión | Dirección | Imagen mental | Cómo se genera aquí |
|---|---|---|---|
| $\sigma_{xx}$ | normal, axial ($x$) | fibras que se estiran/comprimen al **doblar** | flexión por la carga en la punta; máx. en el empotramiento |
| $\sigma_{yy}$ | normal, transversal ($y$) | material **aplastado/estirado** vertical | local, bajo cargas/apoyos; suele ser pequeña |
| $\tau_{xy}$ | tangencial (cizalla) | **baraja de cartas** deslizando | la tracción cortante de impacto; máx. en el centro |

---

## 7. Tensión de Von Mises: ¿qué valores predicen qué fallo?

La tensión de Von Mises $\sigma_{VM}$ es un **escalar equivalente** que resume
todo el estado tensional en un punto en un único número, comparable directamente
con una propiedad del material medida en un ensayo de tracción simple. Su utilidad
es predecir cuándo un **material dúctil** (acero, aluminio) empieza a fallar.

El umbral de referencia es el **límite elástico** (o de fluencia) del material,
$\sigma_y$ (*yield strength*) — una constante tabulada por material. El criterio
de Von Mises dice:

| Condición | Qué le pasa al material |
|---|---|
| $\sigma_{VM} < \sigma_y$ | **Régimen elástico**: deformación reversible. Al quitar la carga, **recupera** su forma. Diseño seguro. |
| $\sigma_{VM} = \sigma_y$ | **Inicio de plastificación**: el material empieza a fluir (fallo según el criterio). |
| $\sigma_{VM} > \sigma_y$ | **Régimen plástico**: deformación **permanente** (no recupera la forma); con más carga, estricción y **rotura**. |

Valores orientativos de $\sigma_y$ (varían mucho con la aleación/tratamiento):

- **Acero** estructural: $\sigma_y \approx 250\text{–}350\ \text{MPa}$ (aceros de
  alta resistencia llegan a $>1000$ MPa).
- **Aluminio**: $\sigma_y \approx 100\text{–}300\ \text{MPa}$ según aleación.

En ingeniería **no** se diseña al límite: se aplica un **coeficiente de seguridad**
$n$ (típico $1.5\text{–}3$), exigiendo $\sigma_{VM} \le \sigma_y/n$. Así se deja
margen frente a incertidumbres (cargas reales, defectos, fatiga).

**Conexión con el proyecto:** el dashboard colorea la malla por $\sigma_{VM}$
([inference.py](../utils/inference.py#L122)) precisamente para localizar de un
vistazo **dónde** el material está más cerca de su límite — típicamente el
**empotramiento**, donde el momento flector y por tanto $\sigma_{xx}$ son máximos.
Una extensión natural del TFM sería comparar el $\sigma_{VM}$ máximo simulado
contra $\sigma_y$ del material para emitir un veredicto automático de
"seguro / plastifica".

> ⚠️ Matiz honesto para defensa: Von Mises es un criterio para materiales
> **dúctiles**. Para **frágiles** (fundición, cerámica, hormigón) se usan otros
> (tensión normal máxima, Mohr-Coulomb), porque esos fallan por tracción, no por
> cortante.

---

### ✅ Resumen de las 7 dudas
1. Campo = función que da $(u,v)$ en **cada** punto e instante; "componente" $u$ vs "campo" completo.
2. Poisson = contracción lateral al estirar; entra en las ecuaciones vía $\lambda$ y las acopla.
3. Euler-Bernoulli = modelo 1D (ODE) que desprecia el cortante; el 2D lo resuelve fielmente.
4. Elastodinámica lineal = elástico + con inercia/tiempo + proporcional; tensión plana por viga delgada de caras libres.
5. $\varepsilon_{xy}$ = distorsión del ángulo recto → genera $\tau_{xy}$; linealizar por desplazamientos diminutos (fiel + manejable + convergencia).
6. $\sigma_{xx}$ = flexión (tracción arriba/compresión abajo, no carga axial); $\sigma_{yy}$ = aplastamiento vertical; $\tau_{xy}$ = capas deslizando (baraja).
7. $\sigma_{VM} < \sigma_y$ elástico (reversible); $\ge \sigma_y$ plástico (permanente)→rotura; diseñar con coef. de seguridad.

---
---

# Parte II — Inferencia, visualización y validación

> Segunda tanda de dudas, surgidas al ejecutar `app.py`. Misma profundidad.

---

## 8. ¿Qué son $u_{xx}$, $u_{yy}$, $u_{xy}$, $u_{tt}$? Derivadas segundas y su sentido físico

Sí: la notación con subíndices es la forma compacta de escribir **derivadas
parciales**. Cada subíndice es "deriva respecto a esta variable", y repetir
subíndice significa **derivar dos veces**:

$$u_x = \frac{\partial u}{\partial x}, \qquad
  u_{xx} = \frac{\partial^2 u}{\partial x^2}, \qquad
  u_{yy} = \frac{\partial^2 u}{\partial y^2}, \qquad
  u_{xy} = \frac{\partial^2 u}{\partial x\,\partial y}, \qquad
  u_{tt} = \frac{\partial^2 u}{\partial t^2}$$

Tu intuición es correcta: $u_{xx}$ **es** la derivada segunda de $u$ respecto a
$x$ (dos veces), no "respecto a $x^2$" (eso no significa nada) — pero se escribe
$\partial^2 u/\partial x^2$ porque el operador $\partial/\partial x$ se aplica dos
veces. Veamos qué representa **físicamente** cada una.

### 8.1 Derivadas primeras: pendiente / razón de cambio

- $u_x = \partial u/\partial x$: cuánto cambia el desplazamiento horizontal $u$ al
  avanzar en $x$. **Es exactamente la deformación axial** $\varepsilon_{xx}$
  (estiramiento). Una derivada primera mide **pendiente** o tasa de cambio.
- $u_t = \partial u/\partial t$: cuánto cambia $u$ con el tiempo en un punto fijo →
  **es la velocidad** del material en ese punto.

### 8.2 Derivadas segundas: curvatura y aceleración

Una derivada **segunda** mide cómo cambia la *pendiente*, es decir, la
**curvatura** (en espacio) o la **aceleración** (en tiempo):

| Símbolo | Definición | Significado físico |
|---|---|---|
| $u_{tt}$ | $\partial^2 u/\partial t^2$ | **Aceleración** del material (en $x$). Es el término de **inercia** $\rho\,u_{tt}$ = masa×aceleración de Newton. Sin él no habría dinámica ni ondas. |
| $u_{xx}$ | $\partial^2 u/\partial x^2$ | **Curvatura** del campo $u$ en dirección $x$. Mide cómo varía la deformación axial $\varepsilon_{xx}$ a lo largo de $x$ → ligado al **gradiente de la tensión** $\sigma_{xx}$. |
| $u_{yy}$ | $\partial^2 u/\partial y^2$ | Curvatura de $u$ en dirección $y$: cómo cambia con la altura el desplazamiento horizontal → parte del **cortante** y de la flexión. |
| $u_{xy}$ | $\partial^2 u/\partial x\,\partial y$ | Derivada **cruzada** (primero en $x$, luego en $y$; el orden no importa). Acopla las dos direcciones: cómo cambia con $y$ la pendiente axial. Aparece en el término $(\lambda+\mu)$ que **acopla** las ecuaciones de $u$ y $v$. |

### 8.3 Por qué la EDP necesita justo derivadas **segundas**

La ecuación de Navier-Cauchy **es la 2ª ley de Newton** escrita para un trocito
de material:

$$\underbrace{\rho\,u_{tt}}_{\text{masa}\,\times\,\text{aceleración}}
  \;=\;
  \underbrace{\mu(u_{xx}+u_{yy}) + (\lambda+\mu)\,\partial_x(u_x+v_y)}_{\text{fuerza interna neta}}$$

- El lado izquierdo $\rho\,u_{tt}$ es **inercia** (derivada 2ª en *tiempo*).
- El lado derecho es la **fuerza interna neta**. La tensión es una derivada
  primera del desplazamiento ($\sigma\sim\partial u$); la *fuerza neta* sobre el
  trocito es cómo **varía** esa tensión de un lado a otro, o sea **otra** derivada
  → derivada 2ª en *espacio* ($\partial^2 u$).

> **Idea clave:** la tensión es la derivada 1ª (cuánto se deforma); la fuerza neta
> que mueve el material es la derivada 2ª (cuánto **desequilibra** la tensión de un
> lado respecto al otro). Por eso el residuo de la PINN exige autodiferenciación de
> 2º orden (ver [doc 02](02_PINN_y_DeepLearning.md)).

---

## 9. Las entradas de la red: ¿los parámetros son fijos o varían?

> ⚠️ **Versión simplificada.** Hoy el escenario es **fijo** y la red sólo recibe
> $d_{in}=3$ entradas $(\xi,\eta,\hat t)$; no hay parámetros. Esta sección explica
> la versión **paramétrica** original ($d_{in}=7$), que se conserva como discusión
> del compromiso simplicidad ↔ generalización.

Cada muestra que entraba a la red (versión paramétrica) era el vector
$(\xi,\,\eta,\,\hat t,\;\hat L,\,\hat c,\,\hat p,\,\hat t_{\mathrm{imp}})$. La clave es
distinguir **dos grupos** de entradas:

| Grupo | Entradas | ¿Qué son? |
|---|---|---|
| **Coordenadas** | $\xi,\;\eta,\;\hat t$ | *Dónde* y *cuándo* evalúo la solución (punto del dominio). |
| **Parámetros** | $\hat L,\;\hat c,\;\hat p,\;\hat t_{\mathrm{imp}}$ | *Qué viga / qué carga* estoy resolviendo (geometría e impacto). |

**Durante el entrenamiento NO son fijos: varían.** El muestreo por Latin
Hypercube ([`build_training_sets`](../utils/sampling.py)) sortea cada punto de
colocación con valores **aleatorios** de los cuatro parámetros dentro de sus
rangos ([config.py](../config.py#L122-L126)):

$$\hat L \in [1, 2], \quad \hat c \in [0.05, 0.15], \quad
  \hat p \in [0.5, 1.5], \quad \hat t_{\mathrm{imp}} \in [0.5, 1.5]$$

Así, en vez de aprender la solución de **una** viga, la red aprende **toda la
familia** de soluciones $(u,v)$ en función también de la geometría y la carga.
Esto es lo que la hace **paramétrica**: matemáticamente está aprendiendo
$\;(u,v) = \mathcal{N}_\theta(\xi,\eta,\hat t;\,\hat L,\hat c,\hat p,\hat t_{\mathrm{imp}})$.

**Cómo se elige luego el valor en la simulación:** en inferencia tú **fijas** los
cuatro parámetros a los valores de los sliders (la app los pasa a
[`evaluate_fields`](../utils/inference.py#L75-L79) como columnas constantes
`np.full(...)`), y dejas que **solo varíen las coordenadas** $(\xi,\eta,\hat t)$
sobre la malla. La red devuelve la solución **de esa viga concreta**, sin
reentrenar. Por eso mover un slider y pulsar *Simular* tarda milisegundos: es una
simple evaluación de la función ya aprendida.

> Analogía: no has aprendido "la raíz de 9"; has aprendido **la función raíz**.
> Luego le pides $\sqrt{9}$, $\sqrt{16}$… al instante. Aquí "el número" es la
> tupla $(\hat L,\hat c,\hat p,\hat t_{\mathrm{imp}})$.

---

## 10. ¿Cómo funciona la inferencia? ¿Se evalúa en otros puntos?

**Sí, exactamente.** Entrenamiento e inferencia usan **conjuntos de puntos
distintos**, y eso es totalmente deseable:

| | Entrenamiento | Inferencia (app) |
|---|---|---|
| Puntos | Aleatorios (LHS): 4000 residuo, 1000/frontera, 1500 IC | **Malla regular** $25\times15\times45$ (t, y, x) — ver [`evaluate_fields`](../utils/inference.py#L47-L48) |
| Parámetros | Variados (familia completa) | **Fijos** a los sliders |
| Derivadas | 2º orden (residuo) — pesado | Solo 1º orden (tensiones) — ligero, `create_graph=False` |
| Objetivo | Ajustar pesos $\theta$ minimizando la pérdida física | Solo **evaluar** la red ya entrenada |

La red es una **función continua** definida en *todo* el dominio, no una tabla de
valores en los puntos de entrenamiento. Una vez ajustados los pesos, puedes
preguntarle por **cualquier** $(\xi,\eta,\hat t)$ — coincida o no con un punto de
entrenamiento. La app elige una malla regular y ordenada simplemente porque es
cómoda para dibujar (animación, mapas de color), no porque la red la necesite.

**Esto es justamente la prueba de generalización:** si la red solo "acertara" en
los 4000 puntos de entrenamiento pero fallara en la malla nueva, estaría
**sobreajustando**. Que dé una solución físicamente coherente en una malla
*distinta* y más densa es señal de que ha aprendido la física, no memorizado
puntos. (Más sobre validación en §12.)

---

## 11. Entender la visualización de `app.py`

### 11.1 ¿Por qué la viga ya aparece deformada en $t=0$?

Porque la condición inicial **no se impone de forma exacta, sino "blanda"**
(*soft constraint*). En esta PINN, "reposo en $t=0$" ($u=v=0$) es un **término más
de la pérdida** ([`loss_initial`](../physics/pde_loss.py#L164-L172)), no una
restricción rígida. La red la cumple **aproximadamente**: en $t=0$ quedan
desplazamientos residuales diminutos (del orden del error de entrenamiento, p. ej.
$\sim10^{-3}$ en unidades adimensionales).

Y aquí entra el segundo factor: el **factor de exageración**. Los desplazamientos
reales son micras sobre una viga de un metro — invisibles a escala real. Para que
se vean, la app los **amplifica** ([app.py](../app.py#L240-L242)):

$$\texttt{scale} = 0.18\,\frac{L}{\max|u,v|}$$

Ese factor amplifica **todo**, incluido el pequeño residuo de $t=0$. Resultado: una
imperfección numérica minúscula se ve como una deformación apreciable al inicio.

> **No es un bug físico, es el error de la IC magnificado.** Es, además, un
> **diagnóstico útil**: cuanto más plana esté la viga en $t=0$, mejor entrenada
> está la red (ver §12). El plan de mejora (§13) propone mostrar el residuo de la
> IC como métrica numérica para cuantificarlo.

### 11.2 ¿Cuánto dura la simulación y por qué? (el régimen temporal)

El horizonte es **fijo**: la malla va de $\hat t=0$ a $\hat t=\hat T=100$
([config.py](../config.py), [inference.py](../utils/inference.py)). En tiempo físico
($t=\hat t\cdot T_{\mathrm{ref}}$, $T_{\mathrm{ref}}=L\sqrt{\rho/E}\approx0.193$ ms para
el acero) la ventana dura $\hat T\cdot T_{\mathrm{ref}}\approx$ **19.3 ms**.

¿Por qué 100 y no otro número? Porque el problema tiene **dos escalas de tiempo muy
distintas** y hay que elegir la correcta (ver [doc 05](05_Diagnostico_resultados.md)):

- **Onda elástica:** $T_{\mathrm{ref}}$ *es* este tiempo (la onda cruza la viga en
  $\sim0.29$ ms). Es lo que fija la adimensionalización del término $\rho\ddot u$.
- **Flexión:** el periodo del 1.er modo de la ménsula es $\sim13.5$ ms, ¡$\sim50\times$
  más lento!

Con el valor antiguo ($\hat T=4$, **0.77 ms**) la ventana era el **5.7 % de un periodo
de flexión**: la viga ni flexaba, sólo se veían **ondas** rebotando. Para ver la
**oscilación de flexión** (que es lo interesante y lo que valida Euler-Bernoulli) hace
falta cubrir $\geq1$ periodo: por eso $\hat T=100$ ($\approx1.44$ periodos). El FEM usa
la **misma** ventana, así que la comparación es justa.

La carga es una **rampa lineal** $g(\hat t)=\min(\hat t/\hat t_{\mathrm{ramp}},1)$ con
$\hat t_{\mathrm{ramp}}=20$ (sube en $\sim0.3$ periodos de flexión y se mantiene): excita
la flexión con dinámica visible sin inyectar las ondas de un escalón brusco. No cambia
la **duración** de la ventana, sólo la rapidez con que entra la carga.

### 11.3 ¿Qué es la tensión de Von Mises que colorea la viga?

Es el escalar de la **pregunta 7**: un único número por punto,
$\sigma_{VM}=\sqrt{\sigma_{xx}^2-\sigma_{xx}\sigma_{yy}+\sigma_{yy}^2+3\tau_{xy}^2}$,
que **resume el estado tensional completo** y se compara con el límite elástico
$\sigma_y$. El color (escala *Turbo*, en MPa) indica **dónde el material está más
solicitado**: las zonas cálidas (rojo) son las más cercanas al fallo — típicamente
el **empotramiento**, donde la flexión es máxima. Es el indicador de "¿dónde
sufre la viga?".

### 11.4 ¿Qué son las "barritas" que se mueven?

Cada cuadradito (`marker symbol="square"`) es un **punto material** de la malla:
su posición dibujada es la **posición de reposo más el desplazamiento**,
exagerado ([app.py](../app.py#L245-L246)):

$$x_{\mathrm{dib}} = x + \texttt{scale}\cdot u, \qquad
  y_{\mathrm{dib}} = y + \texttt{scale}\cdot v$$

O sea: **el movimiento de las barritas ES el campo de desplazamientos** $(u,v)$ —
justo el objeto de la pregunta 1, ahora visto en movimiento. Ves cómo cada punto
de la viga vibra alrededor de su posición de equilibrio tras el impacto. El color
de cada barrita es su Von Mises en ese instante. Con un mismo gráfico ves a la vez
**deformación** (posición) y **tensión** (color).

---

## 12. ¿Cómo sé que mi modelo es bueno? ¿Cuál es el *ground truth*?

Es **la** pregunta de tribunal, y tiene una respuesta conceptual importante:

> **En una PINN no hay datos de referencia: el *ground truth* son las propias
> ecuaciones de la física.** El modelo es bueno si **satisface las ecuaciones**
> (residuo ≈ 0) y las **condiciones de contorno e iniciales**, no si reproduce
> unas etiquetas (que no existen).

A diferencia del deep learning supervisado (donde comparas con etiquetas reales),
aquí la "verdad" es la EDP de Navier-Cauchy. Tienes **cuatro niveles** de
validación, de más interno a más externo:

### 12.1 Nivel 1 — Las pérdidas (lo que ya minimiza el entrenamiento)

Si $\mathcal{L}_{\mathrm{res}}$, $\mathcal{L}_{\mathrm{bc}}$ y
$\mathcal{L}_{\mathrm{ic}}$ son **todas pequeñas**, la solución cumple la EDP, el
contorno y el inicio. Si una se queda alta, esa parte falla (p. ej. IC alta →
viga deformada en $t=0$). **Hay que mirar las tres por separado**, no solo la
total: una total baja con la IC alta es engañosa.

### 12.2 Nivel 2 — Comprobaciones físicas (*sanity checks*) en la propia visualización

Cosas que **deben** cumplirse y puedes verificar a ojo en la app:

- **$t=0$:** la viga debe estar **plana** (sin deformar). Si no → IC mal (§11.1).
- **Empotramiento ($x=0$):** ese extremo **no se mueve nunca**. Si vibra → BC mal.
- **Superficies libres:** sin carga aplicada ahí, las tensiones normales/cortantes
  deben anularse en $y=\pm c$.
- **Reposo previo al impacto:** antes de $\hat t_{\mathrm{imp}}$ la viga apenas debe
  moverse; la oscilación arranca **tras** el golpe.
- **Plausibilidad:** desplazamientos en micras, $\sigma_{VM}$ máxima en el
  empotramiento, oscilación que **no diverge** (energía acotada).

### 12.3 Nivel 3 — Contraste con soluciones de referencia

Para un *veredicto cuantitativo* puedes comparar con un *ground truth* externo en
**casos límite** donde sí existe:

- **Estática de Euler-Bernoulli:** la flecha en punta de una ménsula con carga
  $P$ es $\;v_{\mathrm{tip}} = PL^3/(3EI)\;$ con $I=\tfrac{2}{3}c^3$. El **valor
  medio** (tras amortiguarse la oscilación) debería rondar esa cifra.
- **Frecuencia propia:** la oscilación del extremo (gráfica $v(L,0,t)$) debería
  vibrar a la 1ª frecuencia natural de una viga empotrada,
  $f_1 = \tfrac{1.875^2}{2\pi}\sqrt{EI/(\rho A L^4)}$. Medir el periodo en la
  gráfica y compararlo es un test **numérico** fuerte.
- **FEM:** resolver el mismo caso en un solver de elementos finitos (p. ej.
  FEniCS) y comparar campos punto a punto da el error más riguroso.

### 12.4 Nivel 4 — Residuo en malla nueva (generalización)

Evaluar el **residuo de la EDP** en la malla de inferencia (puntos *no* vistos en
entrenamiento, §10): si sigue siendo pequeño, la red **generaliza**; si se
dispara, **sobreajustó** los puntos de colocación.

> **Qué te dice la visualización sobre si el modelo es bueno (resumen):** viga
> plana en $t=0$ ✓, empotramiento quieto ✓, oscilación amortiguada y acotada ✓,
> $\sigma_{VM}$ máxima donde la física lo predice (empotramiento) ✓, y —si se
> añade (§13)— pérdidas finales bajas y flecha/periodo coherentes con
> Euler-Bernoulli. El §13 propone añadir justo estas métricas para no juzgarlo
> "a ojo".

### 12.5 Dudas sobre la validación física: flecha estática, Euler-Bernoulli y frecuencias

> **Contexto (versión actual).** La referencia **principal** del proyecto es ahora
> el **FEM** (mismo problema, misma malla; tarjeta "🏆 PINN vs FEM"). Euler-Bernoulli
> se mantiene como **referencia analítica complementaria** (fórmula cerrada, sin
> ordenador) y es la que aparece en la tarjeta **Nivel 3**. Los conceptos de abajo
> aplican igual.

Estos cuatro números aparecen en la tarjeta **Nivel 3 · Contraste Euler-Bernoulli**
de la app. Aquí qué es cada uno y cómo leer la comparación.

**¿Qué es la *flecha estática*?**
La **flecha** es el desplazamiento vertical del extremo libre de la viga, $v(L,0,t)$.
Tras aplicar la carga, el extremo oscila y, si hubiera amortiguamiento, acabaría
quieto en una posición desplazada: esa posición de equilibrio es la **flecha
estática** (la deformación "permanente" que produce la carga mantenida, sin la
parte vibratoria). En la app la estimamos como el **promedio temporal** de
$v(L,0,t)$, que filtra la oscilación y deja el nivel medio. Es la cifra natural
para comparar con una fórmula de *estática* (sin tiempo).

**¿Qué es *Euler-Bernoulli*?**
Es la teoría clásica de vigas: un modelo **1D simplificado** que describe la
flexión suponiendo que las secciones planas permanecen planas y perpendiculares
a la fibra neutra (desprecia la deformación por cortante). Da **fórmulas
cerradas** —sin ordenador— que sirven de *patrón de oro* aproximado. Para una
ménsula (empotrada en un extremo, libre en el otro) con carga $P$ en la punta:

$$v_{\mathrm{tip}}^{EB} = \frac{P L^3}{3 E I}, \qquad I = \frac{2}{3}c^3 \ \text{(sección rectangular, espesor unidad)}.$$

Nuestro problema es **2D y dinámico** (resuelve Navier-Cauchy completas, con
cortante e inercia), así que Euler-Bernoulli **no es la verdad exacta**, solo una
**referencia de orden de magnitud**: si la PINN acierta la física, su flecha media
debe quedar *cerca* (mismo orden, error de pocas decenas de %), no idéntica.

**¿Qué es la *frecuencia natural* y la *1ª frecuencia natural* de Euler-Bernoulli?**
Una viga, como una cuerda o un diapasón, tiene **modos propios de vibración**:
formas concretas en las que oscila por sí sola tras un golpe, cada una con su
**frecuencia natural**. La **1ª** (la fundamental) es la más baja y la que
domina el movimiento del extremo. Euler-Bernoulli la da también en forma cerrada:

$$f_1^{EB} = \frac{1.875^2}{2\pi}\sqrt{\frac{E I}{\rho A L^4}}, \qquad A = 2c \ \text{(área de la sección, espesor unidad)}.$$

El número $1.875$ es la primera raíz de la ecuación de modos de una ménsula
(un dato tabulado de la teoría). La app mide la **frecuencia PINN** contando
la oscilación de $v(L,0,t)$ (vía FFT / periodo entre picos) y la compara con
$f_1^{EB}$.

**Cómo leer la comparación PINN vs Euler-Bernoulli.**
- Si ambos pares de números (flecha y frecuencia) están en el **mismo orden** con
  error moderado, la PINN está capturando la física de flexión → **buena señal**.
- Si la **flecha PINN es muchísimo menor** que la de Euler-Bernoulli (p. ej. 15 µm
  frente a 800 µm) y la $\sigma_{VM}$ máx. sale en la **punta** en vez del empotramiento,
  no es que Euler-Bernoulli "falle": es que la **red colapsó a la solución casi trivial**
  $u,v\approx0$. Es el fallo intrínseco de las PINN en flexión **controlada por fuerza**
  (la EDP del interior es homogénea → $u\approx0$ da residuo $\approx0$). La solución del
  proyecto es el **equilibrio seccional** (imponer el cortante/momento integral de cada
  sección); ver [doc 04 §12](04_Defensa_del_proyecto.md) y [doc 05](05_Diagnostico_resultados.md).
  Aquí Euler-Bernoulli cumple su papel de **vara de medir** que delata el colapso —pero
  **¡ojo!**: sólo es válido en el **régimen de flexión** ($\hat T$ que cubra $\geq1$
  periodo); en el régimen de ondas la comparación con EB no tiene sentido (doc 05 §1).

> En resumen: Euler-Bernoulli y sus frecuencias **no son el objetivo a igualar**,
> sino un **termómetro barato** (fórmula cerrada, microsegundos) para saber si la
> PINN se mueve en el rango físico correcto o se ha quedado "plana". El **veredicto
> cuantitativo** lo da el **error L2 vs FEM**, no la pérdida total (que puede ser baja
> en un mínimo trivial).

---

## 13. Plan de mejora de la visualización

> ✅ **IMPLEMENTADO.** Lo que sigue fue la hoja de ruta y hoy está en el código:
> métricas de entrenamiento (`train.py` + tarjeta en `app.py`), métricas de
> simulación y validación física (`utils/metrics.py`), y el pulido de la app. La
> única acción pendiente del lado del usuario es **reentrenar una vez** para
> regenerar el checkpoint con los campos nuevos (`final_comps`, `history`, …).
> Los niveles 3 y 4 de validación son **conmutables** desde `config.VALIDACION`.

### 13.1 Exponer métricas de entrenamiento

**Problema actual:** el checkpoint solo guarda `best_loss`
([train.py](../train.py#L210-L216)); se pierden las componentes
($\mathcal{L}_{\mathrm{res}}$, $\mathcal{L}_{\mathrm{bc}}$,
$\mathcal{L}_{\mathrm{ic}}$) y la curva de convergencia (`history`).

**Plan:**
1. **`train.py`** — ampliar el `payload` guardado con:
   - `final_comps`: dict `{res, bc, ic}` de la última (mejor) evaluación.
   - `history`: la lista `history` ya existente (loss y componentes por época) —
     submuestreada (p. ej. 1 de cada N) para no inflar el archivo.
   - `n_params`, `adam_epochs`, `lbfgs_epochs`, `material_entrenamiento` (ya está).
2. **`utils/inference.py`** (o un `load_model` extendido) — devolver esos metadatos
   junto al modelo.
3. **`app.py`** — nueva **tarjeta "Métricas de entrenamiento"** (panel lateral o
   pie) que muestre:
   - Pérdida total final y **desglose** $\mathcal{L}_{\mathrm{res}},
     \mathcal{L}_{\mathrm{bc}}, \mathcal{L}_{\mathrm{ic}}$ (con código de color
     verde/ámbar/rojo según umbral).
   - **Curva de convergencia** (loss vs. época, escala log) como `dcc.Graph`.
   - Nº de parámetros, épocas Adam/L-BFGS, material de entrenamiento.

### 13.2 Métricas numéricas de la simulación

**Problema actual:** las gráficas son cualitativas; faltan **números**.

**Plan:** en `evaluate_fields` ya se tienen los campos; calcular y devolver un dict
`metrics` con escalares por simulación, y mostrarlo en una **tarjeta "Resultados"**:

| Métrica | Fórmula / origen | Unidad |
|---|---|---|
| Flecha máx. del extremo | $\max_t \lvert v(L,0,t)\rvert$ | µm |
| Flecha estática (media tras oscilar) | media de $v_{\mathrm{tip}}$ en la cola | µm |
| Periodo / frecuencia de oscilación | de los picos de $v(L,0,t)$ (FFT o cruces) | ms / kHz |
| Desplazamiento máx. global | $\max\lVert(u,v)\rVert$ | µm |
| $\sigma_{xx}$ máx. (y dónde) | $\max\lvert\sigma_{xx}\rvert$ + posición | MPa, (x,y) |
| **Von Mises máx.** (y dónde) | $\max\sigma_{VM}$ + posición | MPa, (x,y) |
| Deformación máx. $\varepsilon$ | de los gradientes ya calculados | — (×10⁻⁶) |
| **Factor de seguridad** | $\sigma_y/\max\sigma_{VM}$ + veredicto seguro/plastifica | — |

El **factor de seguridad** cierra el círculo con la pregunta 7: comparar
$\max\sigma_{VM}$ con $\sigma_y$ del material y emitir un veredicto automático
("✅ elástico" / "⚠️ plastifica").

### 13.3 Validación física automática (opcional, alto valor de tribunal)

- Mostrar el **residuo medio de la EDP** evaluado en la malla de inferencia
  (requiere derivadas 2º orden; calcularlo en un subconjunto pequeño de puntos
  para no penalizar la latencia) → métrica directa de "cumple la física" (§12.4).
- Comparar la **flecha estática** y la **frecuencia** medidas contra las fórmulas
  de Euler-Bernoulli (§12.3) y mostrar el **% de error** como sello de calidad.
- Resaltar numéricamente el **residuo de la IC** en $t=0$ (cuantifica §11.1).

### 13.4 Pulido profesional de la app

- **Eje temporal claro:** etiquetar "ms" sin ambigüedad y, opcionalmente, mostrar
  también $\hat t$ adimensional; marcar en la gráfica de oscilación el instante
  $\hat t_{\mathrm{imp}}$ con una línea vertical.
- **Factor de exageración visible:** indicar en pantalla el `scale` aplicado (p. ej.
  "deformación ×3.2·10⁴") para que el usuario sepa que está magnificada.
- **Indicador de calidad del modelo:** un semáforo en cabecera basado en las
  pérdidas finales (§13.1) — verde si las tres están por debajo de su umbral.
- Curva de **energía** o de **σ_VM máx. vs. tiempo** como gráfica adicional.

> **Orden sugerido de implementación:** 13.1 (guardar métricas en `train.py` —
> requiere **reentrenar** una vez para regenerar el checkpoint con los nuevos
> campos) → 13.2 (métricas de simulación, solo tocan `inference.py`/`app.py`, sin
> reentrenar) → 13.3/13.4 (refinos). Nada de esto se ha ejecutado todavía.

---

### ✅ Resumen de la Parte II
8. $u_{xx},u_{yy},u_{xy},u_{tt}$ = derivadas **segundas** ($\partial^2/\partial x^2$, etc.); espaciales = **curvatura**→fuerza interna, temporal $u_{tt}$ = **aceleración**/inercia. La EDP es Newton: inercia (2ª en t) = fuerza neta (2ª en espacio).
9. Las 7 entradas = 3 coordenadas + 4 parámetros; en **entrenamiento varían** (LHS, familia de vigas), en **inferencia se fijan** a los sliders → red **paramétrica**, sin reentrenar.
10. Inferencia = **evaluar** la red (función continua) en una **malla nueva regular**, parámetros fijos, solo derivadas 1º orden. Puntos ≠ entrenamiento → prueba de generalización.
11. Visualización: deformada en $t=0$ = **IC blanda + exageración** (no bug); ventana fija de **19.3 ms** ($\hat T=100$, régimen de **flexión**, $\geq1$ periodo); color = Von Mises (dónde sufre); barritas = campo $(u,v)$ en movimiento.
12. Sin *ground truth*: la verdad **es la física** (residuo≈0). Validar por pérdidas desglosadas, *sanity checks* visuales, contraste con Euler-Bernoulli/FEM y residuo en malla nueva.
13. Plan (no ejecutado): guardar componentes+histórico en checkpoint → tarjeta de métricas de entrenamiento; métricas numéricas de simulación (flecha, periodo, σ_VM máx, factor de seguridad); validación física automática; pulido de la app.
