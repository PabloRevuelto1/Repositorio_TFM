---
name: pinn-debugging
description: Diagnosticar y arreglar problemas de entrenamiento de PINNs (Physics-Informed Neural Networks) en el proyecto viga2D y similares. Úsala cuando una PINN no converge, colapsa a la solución trivial, la pérdida se estanca, las tensiones aparecen en el sitio equivocado, o hay que rebalancear pesos/restricciones. Cubre colapso trivial, condicionamiento del residuo, restricciones duras vs blandas y balance de pérdidas.
---

# Depuración de PINNs

## Síntomas → causa → fix (catálogo)

### 1. La red colapsa a la solución trivial (u,v ≈ 0)
**Síntomas**: pérdida estancada en un valor no nulo; desplazamientos órdenes de
magnitud por debajo de lo físico; tensión/deformación casi nula salvo donde se
*impone* directamente la carga.
**Causa**: las condiciones homogéneas (empotramiento, IC, superficies libres) se
satisfacen exactamente con u=v=0, y la única condición inhomogénea (la carga) está
diluida en un peso compartido → el mínimo trivial es un atractor.
**Fix (orden de impacto)**:
1. **Restricciones duras (hard constraints)**: multiplica la salida de la red por un
   ansatz que se anule donde la condición es homogénea. Para IC en t=0 con velocidad
   nula y empotramiento en x=0: `salida = ξ · t̂²/(t̂²+τ²) · Ñ(...)`. Cerca de t̂=0 va
   como `(t̂/τ)²` → anula desplazamiento *y* velocidad; `ξ` anula en ξ=0. La envolvente
   temporal **SATURA a 1** (no crece): usar `(t̂/T̂)²` la hace crecer ×T̂² sobre
   horizontes largos y descondiciona (regla: la envolvente del hard-constraint nunca
   debe crecer sin cota sobre el dominio).
2. **Peso propio para la condición motora** (la carga): sácala del peso compartido y
   dale un peso mayor (en viga2D `w_load=20` vs `w_bc=5`).
3. Verifica el ansatz con un forward de 4 puntos: debe dar 0 en ξ=0 y en t̂=0, y ≠0
   en el interior.

### 2. Magnitud del residuo desequilibrada frente a las demás pérdidas
**Síntomas**: el residuo baja pero la solución es físicamente pobre; la EDP queda
infra- o sobre-impuesta respecto al contorno.
**Causa**: el `mean(r²)` puede salir de un orden muy distinto a `load`/`bc` por las
escalas de la EDP adimensional, descompensando el gradiente.
**Fix**: **reescala el residuo por un factor constante** para que `mean(r²)` quede del
mismo orden que las demás pérdidas (en viga2D `r·ĉ`, ĉ~0.1). **OJO**: un factor
constante NO rebalancea los términos INTERNOS del residuo (`U_ηη/ĉ²` vs `U_t̂t̂` siguen
con su ratio intrínseco ~100×); sólo fija la MAGNITUD GLOBAL. Por eso `r·ĉ²` (probado
antes) era un error: hundía `mean(r²)` a ~1e-4 de lo natural e infra-imponía la EDP.
El rebalanceo INTERNO real, si hace falta, es término a término o con pesos adaptativos
(NTK / grad-norm). Ajusta `w_res` para el balance fino.

### 2-bis. Colapso trivial pese a hard-constraints (problema CONTROLADO POR FUERZA)
**Síntomas**: pérdida total bajísima (p.ej. 2e-5) pero la solución es casi nula; flecha
~100× corta; tensión sólo donde se impone la carga (la punta), no en el empotramiento;
L2 vs FEM ~99 %. El semáforo de pérdida dice "BUENO" pero el resultado es pésimo.
**Causa**: la EDP del interior es **homogénea** (ρü=∇·σ, sin fuerza de volumen) → `u≈0`
la satisface con residuo≈0. Si lo único que fuerza la respuesta global es una BC de
Neumann **débil** (una tracción pequeña), el gradiente se queda en la cuenca trivial. La
pérdida total NO lo ve (mínimo degenerado). Es intrínseco a las PINN *vanilla* en
elasticidad accionada por fuerza (la FEM no lo sufre: su K global acopla todo).
**Fix (física pura)**: impón el **equilibrio INTEGRAL** que el residuo local no transmite.
Para una ménsula con carga en la punta, en toda sección ξ: `∫τ̂_xy dη=-K_shear·g` y
`∫σ̂_xx·η dη=+K_moment·(1-ξ)·g` (consecuencias de ∇·σ=0, NO datos; verifica signo/forma
contra el FEM). Con `u≈0` el momento integral da 0 ≠ target → saca del trivial. Imponlo
BLANDO (no rigidices a cuasi-estático: la dinámica la aporta el residuo). En viga2D:
`pde_loss.loss_equilibrium`, `sampling.sample_sections`, `w_eq`. **Alternativas** si no
basta: ansatz aditivo con solución particular (teoría guiada), o control por
desplazamiento (prescribir y hard-constrain el movimiento en vez de una fuerza débil).

### 3. La pérdida no baja desde el inicio
- Comprueba la **adimensionalización**: en SI el residuo se escala ~1e11 y no
  converge. Todo debe estar en variables adimensionales de orden unidad.
- Comprueba la **normalización de entradas** a ~[-1,1] (buffers lo/hi).
- LR demasiado alto / activaciones saturadas.

### 4. NaN en pérdidas o métricas
- A menudo NO es física: revisa que se esté cargando el **checkpoint correcto**
  (no uno antiguo/incompatible) y que las claves del payload existan.
- Divisiones por escalas nulas, `sqrt` de negativos (usa `clip(...,0,None)` en Von Mises).

## Principios de balance de pérdidas
- Cada componente debe contribuir gradientes de magnitud comparable. Si una domina,
  rebalancea pesos o reescala el término, no subas ciegamente épocas.
- Con restricciones duras, las componentes que pasan a ser exactas (IC, empotramiento)
  quedan ≈0 y sus pesos son casi irrelevantes: úsalas como *monitores*.
- Doble optimizador: **Adam** (explora) → **L-BFGS** (refina alta precisión).

## Cómo validar SIN romper WSL
- `python -m py_compile` siempre.
- Lotes diminutos (≤16 pts) en CPU para probar forward/loss/backward.
- La 2.ª derivada sobre miles de puntos es la operación pesada → solo en GPU/RunPod.

## Validar que el modelo es bueno
- Comparación con **ground truth FEM** (referencia objetiva): error L2 de campos.
- Sanity checks: plano en t=0, empotramiento quieto, superficies descargadas.
- Contraste analítico Euler-Bernoulli (flecha, frecuencia) como referencia cerrada.
- σ_VM máxima debe estar en el **empotramiento** (momento flector máx), no en la punta.
