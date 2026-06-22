# 📘 Guía Maestra del TFM — Viga 2D con PINNsFormer

Esta carpeta `docs/` contiene la explicación didáctica y profunda del proyecto,
pensada para que domines **todo** lo necesario para defenderlo como experto.

## Mapa de lectura

| Documento | Qué responde |
|---|---|
| **[01 — El sistema físico y la elasticidad](01_Sistema_y_Elasticidad.md)** | ¿Qué se resuelve (PDE/ODE)? ¿Por qué? Todos los conceptos de elasticidad: campo de desplazamientos, deformaciones, tensiones, cortantes/flectores, parámetros del material, condiciones de contorno y cómo/dónde/qué efecto tiene la carga. |
| **[02 — Cómo lo resuelve la red (PINN + Deep Learning)](02_PINN_y_DeepLearning.md)** | ¿Cómo se resuelve? Derivadas, autodiferenciación, por qué 2.º orden, la función de pérdida y el backprop. Arquitectura PINNsFormer completa, entradas/salidas, adimensionalización, sobre qué generaliza y ventajas frente a Elementos Finitos. |
| **[03 — Respuestas a dudas concretas](03_Respuestas_dudas.md)** | Dudas puntuales en dos partes: (I) campo de desplazamientos, Poisson, Euler-Bernoulli, tensión plana, cortante, σ_xx/σ_yy/τ_xy, Von Mises; (II) derivadas segundas, entradas de la red, inferencia, la app, validación sin ground truth y el plan de mejora de la visualización. |
| **[04 — Defensa del proyecto](04_Defensa_del_proyecto.md)** | Guion integral para tribunal: el proyecto entero (problema, matemáticas, PINN, arquitectura, MLOps, app, validación) a nivel de estudiante de máster, **incluyendo un análisis honesto de los resultados actuales y sus limitaciones**. |
| **[05 — Diagnóstico honesto de los resultados](05_Diagnostico_resultados.md)** | Por qué los resultados salen malos: las **dos escalas mal elegidas** (tiempo de onda vs flexión; desplazamiento ×225), por qué los checks de Euler-Bernoulli/frecuencia son comparaciones inválidas, **cómo sería un resultado bueno y por qué**, y el camino a seguir (régimen de flexión vs de ondas). |
| **[BITÁCORA — Historial del proyecto](BITACORA.md)** | Memoria cronológica persistente (desde el inicio hasta ahora) que se actualiza con cada cambio. Para el estado actual resumido y las reglas de trabajo, ver [`../CLAUDE.md`](../CLAUDE.md) (se auto-carga en cada sesión de Claude Code). |

## Resumen ejecutivo en 10 líneas

- **Problema:** elastodinámica 2D de una **viga empotrada** que recibe una **carga de impacto** cortante en su extremo libre. Se quiere conocer cómo vibra y se tensiona en el tiempo.
- **Modelo matemático:** un sistema de **EDPs** (ecuaciones de Navier-Cauchy en tensión plana) — derivadas en *espacio* (x, y) **y** *tiempo* (t).
- **Método numérico:** una **PINN** (Physics-Informed Neural Network). Una red neuronal aprende la solución haciendo que el **residuo de la EDP** sea ≈0 en miles de puntos, sin datos de referencia: la física *es* la supervisión.
- **Arquitectura concreta:** **PINNsFormer** (un Transformer; Zhao et al., ICLR 2024) que procesa una *pseudo-secuencia temporal* de cada punto.
- **Truco clave:** todo se resuelve en **variables adimensionales**, porque con unidades del SI (E≈2·10¹¹ Pa, u≈10⁻⁶ m) la red no converge.
- **Versión simplificada (feedback de tutores):** se resuelve **un escenario fijo** ($d_{in}=3$: $\xi,\eta,\hat t$) en vez de una red paramétrica; geometría/carga constantes (`config.VIGA`). La carga es una **rampa lineal** (antes sigmoide).
- **Ground truth real:** el mismo problema se resuelve por **Elementos Finitos** (`scikit-fem`, integración de Newmark) sobre la misma malla e instantes, y la app **compara PINN vs FEM** —gráfica y métricamente— declarando el **ganador por categoría** (precisión, tiempo, ubicación de la tensión máxima…).

> Las cifras y fórmulas de estos documentos están tomadas directamente del código
> ([config.py](../config.py), [physics/pde_loss.py](../physics/pde_loss.py),
> [models/pinnsformer.py](../models/pinnsformer.py)), no de fuentes externas.
