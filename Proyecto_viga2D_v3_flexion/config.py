"""
config.py
=========
Configuración central del proyecto de elastodinámica 2D con PINNsFormer.

Aquí se centralizan TODAS las constantes físicas, las escalas de
adimensionalización, los rangos del dominio paramétrico y los
hiperparámetros del modelo y del entrenamiento.

------------------------------------------------------------------------
NOTA SOBRE LA ADIMENSIONALIZACIÓN (clave para que la PINN entrene)
------------------------------------------------------------------------
Trabajar directamente con unidades del SI es inviable para una PINN:
el acero tiene E = 210 GPa (~1e11) y los desplazamientos son del orden
de micras (~1e-6 m). Los residuos de la EDP quedarían escalados por
factores de 1e11 y la red no converge.

Por ello todo el problema se resuelve en variables adimensionales:

    ξ = x / L            ∈ [0, 1]        (coordenada axial normalizada)
    η = y / c            ∈ [-1, 1]       (coordenada transversal normalizada)
    t̂ = t / T_REF        ∈ [0, T_HAT]    (tiempo normalizado por T_REF)
    U = u / U_REF,  V = v / U_REF         (desplazamientos normalizados)

con las escalas de referencia:

    T_REF(material) = L_REF * sqrt(ρ / E)      (tiempo de onda elástica)
    σ_SCALE         = 3 * P_REF / (4 * C_REF)  (escala de tensión, ∝ carga)
    U_REF(material) = L_REF * σ_SCALE / E       (escala de desplazamiento)

Sustituyendo en las ecuaciones de Navier-Cauchy, los coeficientes
adimensionales quedan de orden unidad:

    μ̂ = 1 / (2 (1 + ν)),     λ̂ = ν / (1 - ν²)   [tensión plana]

y el residuo adimensional es (ver physics/pde_loss.py):

    μ̂(U_ξξ/L̂² + U_ηη/ĉ²) + (λ̂+μ̂)(U_ξξ/L̂² + V_ξη/(L̂ĉ)) - U_t̂t̂ = 0

La red se entrena con el coeficiente de Poisson del ACERO; al ser ν muy
parecido entre acero (0.30) y aluminio (0.33) la solución adimensional es
prácticamente la misma y el selector de material de la app reescala los
resultados a unidades físicas mediante U_REF y T_REF de cada material.
"""

from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# 1. ESCALAS DE REFERENCIA (adimensionalización)
# ----------------------------------------------------------------------
L_REF = 1.0        # [m]   longitud de referencia
C_REF = 0.1        # [m]   semi-altura de referencia (peralto/2)
P_REF = 1.0e5      # [N/m] carga cortante total de referencia (por unidad de espesor)

# Momento de inercia de la sección de referencia (espesor unidad): I = 2c³/3.
I_REF = 2.0 * C_REF ** 3 / 3.0

# ----------------------------------------------------------------------
# ESCALA DE TENSIÓN — RÉGIMEN DE FLEXIÓN  (ver docs/05)
# ----------------------------------------------------------------------
# Antes se adimensionalizaba por la TRACCIÓN aplicada σ=3P/4c (≈0.75 MPa). Pero la
# viga trabaja a FLEXIÓN: una tracción cortante pequeña en la punta genera, por el
# brazo de palanca (L/c), una tensión de flexión ~30× mayor en el empotramiento, y
# una flecha ~225× la escala de desplazamiento. Con aquella escala el desplazamiento
# adimensional valía ~225 (debería ser O(1)) y la red no podía alcanzarlo.
#
# Se adimensiona por la TENSIÓN DE FLEXIÓN de la ménsula de referencia,
#     Σ = M_REF·C_REF/I_REF = (P_REF·L_REF)·C_REF/I_REF = P_REF·L_REF²/(2·C_REF³),
# equivalente a Σ = E·U_REF/L_REF con U_REF = flecha estática de referencia. Así el
# desplazamiento adimensional es O(1), la tensión de flexión es O(1) y la tracción
# aplicada queda dimensionalmente PEQUEÑA (~0.015), lo cual es físicamente correcto.
# Independiente del material (no depende de E).
SIGMA_SCALE = P_REF * L_REF ** 2 / (2.0 * C_REF ** 3)   # [Pa]  (tensión de flexión ref.)

# Semi-altura de referencia adimensional
C_HAT_REF = C_REF / L_REF


# ----------------------------------------------------------------------
# 2. MATERIALES
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Material:
    nombre: str
    E: float          # módulo de Young [Pa]
    nu: float         # coeficiente de Poisson [-]
    rho: float        # densidad [kg/m³]
    sigma_y: float    # límite elástico (yield strength) [Pa] — umbral de plastificación

    # --- Constantes derivadas (tensión plana) ---
    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        return self.E * self.nu / (1.0 - self.nu ** 2)

    @property
    def mu_hat(self) -> float:
        """μ̂ = μ / E = 1 / (2(1+ν))  (coeficiente adimensional)."""
        return 1.0 / (2.0 * (1.0 + self.nu))

    @property
    def lam_hat(self) -> float:
        """λ̂ = λ / E = ν / (1-ν²)  (coeficiente adimensional, tensión plana)."""
        return self.nu / (1.0 - self.nu ** 2)

    @property
    def T_ref(self) -> float:
        """Tiempo de referencia T_REF = L_REF * sqrt(ρ/E) [s]."""
        return L_REF * (self.rho / self.E) ** 0.5

    @property
    def U_ref(self) -> float:
        """Desplazamiento de referencia U_REF = L_REF * σ_SCALE / E [m]."""
        return L_REF * SIGMA_SCALE / self.E


MATERIALES = {
    # Proyecto SIMPLIFICADO: un único material (acero estructural S275).
    # sigma_y orientativo: acero estructural S275 ≈ 275 MPa.
    "Acero":    Material("Acero",    E=210e9, nu=0.30, rho=7850.0, sigma_y=275e6),
}

# Material usado para entrenar la red (define μ̂, λ̂ de la EDP).
MATERIAL_ENTRENAMIENTO = "Acero"
MATERIAL = MATERIALES[MATERIAL_ENTRENAMIENTO]   # acceso directo (único material)


# ----------------------------------------------------------------------
# 2-bis. ESCENARIO FIJO DE LA VIGA  (proyecto simplificado)
# ----------------------------------------------------------------------
# Siguiendo el feedback de los tutores, el proyecto se SIMPLIFICA: en vez de una
# red paramétrica que generaliza a cualquier geometría/carga, se resuelve UN
# único caso fijo. La geometría, la carga y el perfil temporal del impacto son
# constantes; la red ya solo recibe (ξ, η, t̂) como entradas (d_in = 3).
@dataclass(frozen=True)
class Viga:
    L: float = 1.5         # [m]    longitud de la viga
    c: float = 0.10        # [m]    semi-altura (peralto/2)
    P_max: float = 1.0e5   # [N/m]  carga cortante máxima (por unidad de espesor)

    # Perfil temporal de la carga: RAMPA LINEAL que sube de 0 a P_max en el
    # intervalo t̂ ∈ [0, t_ramp_hat] y se mantiene constante después.
    #     g(t̂) = min(t̂ / t_ramp_hat, 1)          (suave, arranca exactamente en 0)
    # RÉGIMEN DE FLEXIÓN (docs/05): la rampa dura ~0.3 periodos de flexión
    # (T_bending/T_REF ≈ 70 en unidades t̂), de modo que se excita el modo de
    # flexión con dinámica visible (sobreoscilación moderada) sin inyectar las
    # ondas de alta frecuencia de un escalón brusco. Antes valía 1.0 (= tiempo de
    # ONDA), que era un escalón instantáneo frente a la flexión.
    t_ramp_hat: float = 20.0

    # --- Adimensionales derivados (con las escalas de referencia de arriba) ---
    @property
    def L_hat(self) -> float:
        return self.L / L_REF

    @property
    def c_hat(self) -> float:
        return self.c / L_REF

    @property
    def p_hat(self) -> float:
        return self.P_max / P_REF

    @property
    def tau_hat_peak(self) -> float:
        """Pico adimensional de la tracción cortante aplicada en ξ=1:
        τ̂_peak = (3·P_max/4c) / SIGMA_SCALE. Pequeño (~0.015) en el régimen de
        flexión: la tracción motora es mucho menor que la tensión de flexión que
        induce. Sustituye al antiguo atajo p̂·(ĉ_REF/ĉ), válido sólo con la escala
        de tensión anterior (= tracción aplicada)."""
        return (3.0 * self.P_max / (4.0 * self.c)) / SIGMA_SCALE

    @property
    def K_shear(self) -> float:
        """Cortante adimensional resultante por sección: V̂(ξ)=∫τ̂_xy dη = -K_shear·g(t̂).
        Constante a lo largo de la viga (carga sólo en la punta). K_shear=P/(Σ·c)."""
        return self.P_max / (SIGMA_SCALE * self.c)

    @property
    def K_moment(self) -> float:
        """Momento flector adimensional por sección: M̂(ξ)=∫σ̂_xx·η dη = +K_moment·(1-ξ)·g(t̂).
        Lineal: máximo en el empotramiento (ξ=0), nulo en la punta. K_moment=P·L/(Σ·c²).
        Signo y forma verificados contra el FEM (ground truth)."""
        return self.P_max * self.L / (SIGMA_SCALE * self.c ** 2)


VIGA = Viga()


# ----------------------------------------------------------------------
# 3. DOMINIO (rangos adimensionales de muestreo)
# ----------------------------------------------------------------------
# Al fijarse la geometría/carga (escenario VIGA), el dominio de muestreo es solo
# espacio-temporal: (ξ, η, t̂). Ya no hay rangos paramétricos.
@dataclass(frozen=True)
class Dominio:
    xi: tuple = (0.0, 1.0)        # ξ = x / L
    eta: tuple = (-1.0, 1.0)      # η = y / c
    # Horizonte temporal t̂ ∈ [0, T_hat]. RÉGIMEN DE FLEXIÓN (docs/05): debe cubrir
    # ≥1 periodo de flexión. T_bending/T_REF ≈ 70 (T_REF es el tiempo de ONDA, ~50×
    # más rápido que la flexión), así que T_hat=100 ≈ 1.44 periodos (T_phys≈19 ms):
    # una oscilación clara, resoluble por FFT y aún suave para la red. Antes era 4.0
    # (= 0.77 ms = 5.7 % de un periodo → la viga ni flectaba; sólo se veían ondas).
    T_hat: float = 100.0

    @property
    def t(self) -> tuple:
        return (0.0, self.T_hat)


DOMINIO = Dominio()


# ----------------------------------------------------------------------
# 4. HIPERPARÁMETROS DEL MODELO PINNsFormer
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModeloCfg:
    d_in: int = 3        # (ξ, η, t̂)  — escenario fijo, sin entradas paramétricas
    d_out: int = 2       # (U, V)  desplazamientos adimensionales
    d_model: int = 32    # dimensión del embedding (paper)
    d_hidden: int = 512  # capa oculta de la cabeza de salida
    N: int = 1           # nº de capas encoder/decoder (paper)
    heads: int = 2       # nº de cabezas de atención (paper)

    # Pseudo-secuencia temporal (truco PINNsFormer)
    k: int = 5           # longitud de la secuencia (num_step)
    seq_step: float = 1e-3   # Δt̂ entre pseudo-instantes

    # --- RESTRICCIONES DURAS de la condición inicial y del empotramiento ---
    # En lugar de imponer t̂=0 y ξ=0 como términos "blandos" de la pérdida (que la
    # red tiende a satisfacer colapsando a la solución trivial u=v=0), se multiplica
    # la salida del transformer por un factor-ansatz  ξ · t̂²/(t̂²+τ²)  que SE ANULA
    # exactamente en el empotramiento (ξ=0 → U=V=0) y en el instante inicial
    # (t̂=0 → U=V=0 y, por ser ~t̂² cerca de 0, también la velocidad U_t̂=V_t̂=0).
    # La envolvente temporal SATURA a 1 para t̂≫τ (no crece): antes era (t̂/T_hat)²,
    # que sobre el horizonte largo de flexión crecía ×(T_hat)²≈1e4 y descondicionaba
    # el problema. NO altera la arquitectura PINNsFormer del paper.
    hard_constraints: bool = True
    # τ de subida de la envolvente de la CI (en unidades t̂). Se elige ≈ t_ramp_hat
    # (la envolvente "se enciende" al ritmo de la carga). Ver models/pinnsformer.py.
    ic_tau: float = 20.0


MODELO = ModeloCfg()


# ----------------------------------------------------------------------
# 5. HIPERPARÁMETROS DE ENTRENAMIENTO
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EntrenamientoCfg:
    # Nº de puntos de colocación (Latin Hypercube Sampling).
    # El horizonte temporal del régimen de flexión es 25× mayor (T_hat 4→100), así
    # que se sube n_res para mantener densidad de muestreo en (ξ,η,t̂).
    n_res: int = 6000     # residuo interior
    n_bc: int = 1000      # cada frontera espacial
    n_ic: int = 1500      # condición inicial

    # EQUILIBRIO SECCIONAL (ver pde_loss.loss_equilibrium): rompe el colapso a la
    # solución trivial imponiendo el cortante/momento integral que la EDP local no
    # transmite (problema controlado por fuerza, EDP homogénea). n_sec pares (ξ,t̂),
    # cada uno con n_eta_eq nodos en η para integrar ∫·dη por sección.
    n_sec: int = 400      # nº de secciones (pares ξ,t̂)
    n_eta_eq: int = 17    # nodos en η por sección (cuadratura trapezoidal)

    # Optimización (algo más larga: la ventana temporal es mayor)
    adam_epochs: int = 3000
    adam_lr: float = 1e-3
    lbfgs_epochs: int = 300
    lbfgs_lr: float = 1.0

    # Pesos de las componentes de la pérdida.
    # Con restricciones DURAS (MODELO.hard_constraints), las condiciones de
    # empotramiento e inicial se cumplen por construcción (≈0), de modo que la
    # CARGA en el extremo libre es la única condición que "tira" de la solución
    # fuera del cero. Por eso recibe su propio peso, MAYOR que el resto, para que
    # el optimizador no la sacrifique frente al residuo/superficies libres.
    w_res: float = 1.0
    w_load: float = 20.0   # contorno cargado ξ=1 (condición motora, inhomogénea)
    w_bc: float = 5.0      # contorno homogéneo (empotramiento + superficies libres)
    w_ic: float = 1.0      # condición inicial (redundante con hard-constraint; monitor)
    w_eq: float = 20.0     # equilibrio seccional (saca de la cuenca trivial; ver docs/05)

    seed: int = 0
    checkpoint: str = "checkpoints/best_model.pth"


ENTRENAMIENTO = EntrenamientoCfg()


# ----------------------------------------------------------------------
# 5-bis. VALIDACIÓN FÍSICA DEL MODELO (visualización / app)
# ----------------------------------------------------------------------
# Controla los cuatro niveles de validación descritos en docs/03 (§12):
#   N1 pérdidas finales · N2 sanity-checks · N3 referencia analítica · N4 residuo.
# Los niveles 1 y 2 son prácticamente gratis y van siempre activos.
# Los niveles 3 y 4 son CONMUTABLES con los booleanos de abajo.
@dataclass(frozen=True)
class ValidacionCfg:
    # --- GROUND TRUTH por ELEMENTOS FINITOS (scikit-fem) ---
    # Resuelve EXACTAMENTE el mismo problema (misma malla, mismos instantes) por
    # FEM y lo compara con la PINN campo a campo y métrica a métrica, declarando
    # un "ganador" en cada categoría. Es la validación de referencia del proyecto.
    fem_ground_truth: bool = True
    fem_substeps: int = 4          # subpasos de Newmark entre instantes de salida

    # --- NIVEL 3: contraste con la solución analítica de Euler-Bernoulli ---
    # Fórmula CERRADA (coste despreciable); referencia analítica complementaria.
    referencia_analitica: bool = True

    # --- NIVEL 4: residuo de la EDP evaluado en la malla de inferencia ---
    # Requiere autodiferenciación de 2.º orden → es la operación PESADA (la que el
    # README desaconseja en WSL). Se evalúa sólo sobre un subconjunto pequeño de
    # puntos y POR BLOQUES (ver utils/metrics.validation_residual) para acotar la
    # RAM y no tumbar la app en CPU/WSL. Súbelo si ejecutas la app en GPU.
    residuo_en_malla: bool = False
    n_residuo_check: int = 400      # nº de puntos para el residuo de validación

    # --- Umbrales del semáforo de calidad (pérdidas finales del checkpoint) ---
    # OJO: la pérdida total NO es el termómetro de calidad (depende de las escalas y
    # de los pesos; ver docs/05 §4). El veredicto real es el ERROR L2 PINN vs FEM.
    # Estos umbrales son sólo un semáforo grueso de convergencia del entrenamiento.
    umbral_loss_ok: float = 1.0e-2      # < ok  → verde
    umbral_loss_warn: float = 1.0e-1    # < warn → ámbar; por encima → rojo

    # Coeficiente de seguridad objetivo para el veredicto de la simulación.
    coef_seguridad: float = 1.5


VALIDACION = ValidacionCfg()


# ----------------------------------------------------------------------
# 6. UTILIDAD: límites de normalización de entrada de la red
# ----------------------------------------------------------------------
def input_bounds():
    """Devuelve (lo, hi) con los límites de las 3 entradas (ξ, η, t̂),
    usados para reescalar la entrada de la red a ~[-1, 1]."""
    d = DOMINIO
    lo = [d.xi[0], d.eta[0], d.t[0]]
    hi = [d.xi[1], d.eta[1], d.t[1]]
    return lo, hi
