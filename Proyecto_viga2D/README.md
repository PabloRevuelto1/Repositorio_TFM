# 🔧 Elastodinámica de Viga 2D con PINNsFormer

Resolución y visualización de la **elastodinámica 2D de una viga empotrada**
bajo carga de impacto, mediante la arquitectura **PINNsFormer**
(Zhao et al., *ICLR 2024*, [arXiv:2307.11833](https://arxiv.org/abs/2307.11833)),
implementada fielmente a partir del repositorio oficial (`TFM/pinnsformer`).

El proyecto incluye una **red paramétrica** que generaliza a distintas
geometrías y cargas sin reentrenar, y un **dashboard interactivo en Dash** con
animación de la malla deformada y gráficas de tensiones en tiempo real.

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

Carga de impacto sigmoidea: $P(t) = P_{\max}\cdot\mathrm{sigmoid}\!\left(\alpha(t - t_{\mathrm{imp}})\right)$,
con perfil parabólico en $y$ (evita singularidades) y rampa suave diferenciable.

### Adimensionalización (clave para la convergencia)

Trabajar en unidades SI es **inviable** para una PINN (acero: $E \approx 2\cdot10^{11}$ Pa,
desplazamientos $\sim10^{-6}$ m). Todo el problema se resuelve en variables
adimensionales (ver `config.py`):

$$\xi = \frac{x}{L} \in [0,1], \qquad \eta = \frac{y}{c} \in [-1,1], \qquad \hat{t} = \frac{t}{T_{\mathrm{ref}}}$$

$$U = \frac{u}{U_{\mathrm{ref}}}, \qquad V = \frac{v}{U_{\mathrm{ref}}}$$

$$T_{\mathrm{ref}} = L_{\mathrm{ref}}\sqrt{\frac{\rho}{E}}, \qquad \sigma_{\mathrm{scale}} = \frac{3\,P_{\mathrm{ref}}}{4\,c_{\mathrm{ref}}}, \qquad U_{\mathrm{ref}} = \frac{L_{\mathrm{ref}}\,\sigma_{\mathrm{scale}}}{E}$$

Así los coeficientes de la EDP quedan de orden unidad
($\hat{\mu} = 1/(2(1+\nu)) \approx 0.385$, $\hat{\lambda} = \nu/(1-\nu^2) \approx 0.33$)
y la red entrena de forma estable. El selector de material de la app **reescala**
la solución adimensional a unidades físicas mediante $U_{\mathrm{ref}}$ y $T_{\mathrm{ref}}$
de cada material.

---

## 2. Arquitectura PINNsFormer

Implementación fiel del repo oficial (`models/pinnsformer.py`):
**WaveAct** (activación de onda $w_1\sin+w_2\cos$) → `FeedForward` →
`EncoderLayer`/`DecoderLayer` (atención multi-cabeza, *cross-attention* en el
decoder) → `Encoder`/`Decoder` → `PINNsformer`.

**Extensiones para PINN paramétrica:**
- Embedding de entrada $d_{\mathrm{in}} = 7$: $(\xi,\,\eta,\,\hat{t},\,\hat{L},\,\hat{c},\,\hat{p},\,\hat{t}_{\mathrm{imp}})$
  → la red generaliza a distintas vigas y cargas.
- Salida $d_{\mathrm{out}} = 2$: $(U, V)$.
- **Pseudo-secuencia temporal** (truco del paper, `utils/sampling.py`): cada
  punto se replica $k=5$ veces desplazando $\hat{t}$; la predicción real es el
  primer token.

Hiperparámetros del paper: $k=5$, $d_{\mathrm{model}}=32$, $\mathrm{heads}=2$, $N=1$.

---

## 3. Estructura del proyecto

```
Proyecto_viga2D/
├── config.py              # Constantes, materiales, escalas, hiperparámetros
├── models/pinnsformer.py  # Arquitectura PINNsFormer (paramétrica)
├── physics/pde_loss.py    # ElastodynamicsLoss (L_res, L_bc, L_ic)
├── utils/
│   ├── sampling.py        # LHS + pseudo-secuencia temporal
│   ├── inference.py       # Inferencia vectorizada por bloques (memory-safe)
│   └── registry.py        # Subida/descarga del modelo a W&B (gratis)
├── train.py               # Entrenamiento Adam + L-BFGS → checkpoints/best_model.pth
├── app.py                 # Dashboard Dash (visualización interactiva)
├── runpod_launch.py       # Lanza el entrenamiento en un Pod GPU de RunPod
├── Dockerfile.train       # Imagen de entrenamiento para RunPod
├── requirements.txt
└── .env.example           # Plantilla de claves (W&B, RunPod)
```

---

## 4. Instalación

```bash
pip install -r requirements.txt
cp .env.example .env        # rellena tus claves
```

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

**Coste estimado:** GPU económica (RTX A4000 ≈ 0.20–0.34 $/h). Un entrenamiento
completo (~5000 Adam + 200 L-BFGS) tarda **< 1 h ⇒ < 0.5 $**. El Pod se
termina automáticamente al acabar (`finally: terminate_pod`), así que **no se
acumulan costes**. El modelo (pocos MB) queda en **W&B (almacenamiento
gratuito)**, por lo que no se paga almacenamiento en RunPod.

### Opción B — GPU local

```bash
python train.py --adam 5000 --lbfgs 200 --device cuda
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

**Interfaz:**
- **Panel de control:** material (Acero/Aluminio), $L$, $c$, $P_{\max}$,
  $\hat{t}_{\mathrm{imp}}$ y botón *Simular Dinámica*.
- **Mapa animado:** malla deformada coloreada por **tensión de Von Mises**, con
  Play/Pause y slider temporal nativos de Plotly (oscilación transitoria).
- **Gráfica 1:** desplazamiento del extremo libre $v(L,0,t)$.
- **Gráfica 2:** $\sigma_{xx}(x)$ a lo largo de la viga en el instante seleccionado.

La inferencia se realiza en **< 500 ms** (típicamente decenas de ms en GPU);
mover sliders y cambiar material **no reentrena**, sólo reevalúa la red.

---

## 7. ¿Por qué este reparto RunPod + W&B + local?

| Tarea | Dónde | Motivo |
|---|---|---|
| Entrenamiento (2.º orden, pesado) | **RunPod GPU** | Imposible en WSL; barato y efímero |
| Almacenamiento del modelo | **W&B (gratis)** | Apagar el Pod ⇒ no pagar disco |
| Visualización (1.er orden, ligera) | **Local / WSL** | Una app Dash 24/7 en RunPod gastaría créditos sin aportar nada |

---

## 8. Revisión crítica del enunciado (mejoras aplicadas)

- **Adimensionalización completa**: imprescindible para que la PINN converja
  con valores de acero; sin ella el residuo se escala por $\sim 10^{11}$.
- **$\hat{t}_{\mathrm{imp}}$ como entrada paramétrica** de la red: permite variarlo desde la
  app sin reentrenar (el enunciado lo pedía como slider; sólo es posible si la
  red lo recibe como parámetro).
- **Material vía reescalado físico** en inferencia ($U_{\mathrm{ref}}$, $T_{\mathrm{ref}}$):
  evita reentrenar por material; $\nu$ de acero y aluminio son casi iguales, por
  lo que la solución adimensional es prácticamente la misma.
- **Inferencia por bloques** (`chunk`): acota la memoria para ejecutar la app en
  equipos modestos.
- **Doble optimizador** Adam → L-BFGS (estándar en PINNs de alta precisión).
- **Pesos de pérdida** ($w_{bc}$, $w_{ic}$) para equilibrar contorno/inicial frente
  al residuo.

---

## 9. Referencias

- Z. Zhao, X. Ding, B. A. Prakash. *PINNsFormer: A Transformer-Based Framework
  for Physics-Informed Neural Networks.* ICLR 2024.
- Raissi, Perdikaris, Karniadakis. *Physics-Informed Neural Networks.* JCP 2019.
