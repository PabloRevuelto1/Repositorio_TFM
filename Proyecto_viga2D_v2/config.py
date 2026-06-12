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

# Escala de tensión, independiente del material (depende sólo de la carga/geometría ref.)
SIGMA_SCALE = 3.0 * P_REF / (4.0 * C_REF)   # [Pa]

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
    # sigma_y orientativo: acero estructural S275 ≈ 275 MPa; aluminio 6061-T6 ≈ 240 MPa.
    "Acero":    Material("Acero",    E=210e9, nu=0.30, rho=7850.0, sigma_y=275e6),
    "Aluminio": Material("Aluminio", E=70e9,  nu=0.33, rho=2700.0, sigma_y=240e6),
}

# Material usado para entrenar la red (define μ̂, λ̂ de la EDP).
MATERIAL_ENTRENAMIENTO = "Acero"


# ----------------------------------------------------------------------
# 3. DOMINIO PARAMÉTRICO (rangos adimensionales de muestreo)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Dominio:
    # Coordenadas normalizadas
    xi: tuple = (0.0, 1.0)        # ξ = x / L
    eta: tuple = (-1.0, 1.0)      # η = y / c
    T_hat: float = 4.0            # horizonte temporal adimensional t̂ ∈ [0, T_hat]

    # Parámetros geométricos / de carga (entradas extra a la red)
    L_hat: tuple = (1.0, 2.0)     # L̂ = L / L_REF   (longitud de viga 1–2 m)
    c_hat: tuple = (0.05, 0.15)   # ĉ = c / L_REF   (semi-altura 5–15 cm)
    p_hat: tuple = (0.5, 1.5)     # p̂ = P_max / P_REF
    t_imp: tuple = (0.5, 1.5)     # t̂_impacto ∈ [0.5, 1.5] (≈ T_hat/4)

    # Rampa sigmoidea de la carga de impacto: P(t) = P_max · sigmoid(α(t̂ - t̂_imp))
    alpha_ramp: float = 8.0       # nitidez de la rampa (en unidades adimensionales)

    @property
    def t(self) -> tuple:
        return (0.0, self.T_hat)


DOMINIO = Dominio()


# ----------------------------------------------------------------------
# 4. HIPERPARÁMETROS DEL MODELO PINNsFormer
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModeloCfg:
    d_in: int = 7        # (ξ, η, t̂, L̂, ĉ, p̂, t̂_imp)
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
    # la salida del transformer por un factor-ansatz  ξ·(t̂/T_hat)²  que SE ANULA
    # exactamente en el empotramiento (ξ=0 → U=V=0) y en el instante inicial
    # (t̂=0 → U=V=0 y, por ser cuadrático, también la velocidad U_t̂=V_t̂=0).
    # Así la única condición inhomogénea que queda es la CARGA en ξ=1, lo que
    # impide el colapso trivial. NO altera la arquitectura PINNsFormer del paper:
    # es una transformación de la salida (hard-constraint BC), práctica estándar.
    hard_constraints: bool = True


MODELO = ModeloCfg()


# ----------------------------------------------------------------------
# 5. HIPERPARÁMETROS DE ENTRENAMIENTO
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EntrenamientoCfg:
    # Nº de puntos de colocación (Latin Hypercube Sampling)
    n_res: int = 4000     # residuo interior
    n_bc: int = 1000      # cada frontera espacial
    n_ic: int = 1500      # condición inicial

    # Optimización
    adam_epochs: int = 2000
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
    # --- NIVEL 3: contraste con la solución analítica de Euler-Bernoulli ---
    # Es una fórmula CERRADA (coste despreciable, microsegundos), pero se deja
    # como booleano para activarla/desactivarla a voluntad, tal y como se pidió.
    referencia_analitica: bool = True

    # --- NIVEL 4: residuo de la EDP evaluado en la malla de inferencia ---
    # Requiere autodiferenciación de 2.º orden → es la operación PESADA (la que el
    # README desaconseja en WSL). Se evalúa sólo sobre un subconjunto pequeño de
    # puntos y POR BLOQUES (ver utils/metrics.validation_residual) para acotar la
    # RAM y no tumbar la app en CPU/WSL. Súbelo si ejecutas la app en GPU.
    residuo_en_malla: bool = False
    n_residuo_check: int = 400      # nº de puntos para el residuo de validación

    # --- Umbrales del semáforo de calidad (pérdidas finales del checkpoint) ---
    umbral_loss_ok: float = 1.0e-3      # < ok  → verde
    umbral_loss_warn: float = 1.0e-2    # < warn → ámbar; por encima → rojo

    # Coeficiente de seguridad objetivo para el veredicto de la simulación.
    coef_seguridad: float = 1.5


VALIDACION = ValidacionCfg()


# ----------------------------------------------------------------------
# 6. UTILIDAD: límites de normalización de entrada de la red
# ----------------------------------------------------------------------
def input_bounds():
    """Devuelve (lo, hi) con los límites de cada una de las 7 entradas,
    usados para reescalar la entrada de la red a ~[-1, 1]."""
    d = DOMINIO
    lo = [d.xi[0],  d.eta[0],  d.t[0],  d.L_hat[0], d.c_hat[0], d.p_hat[0], d.t_imp[0]]
    hi = [d.xi[1],  d.eta[1],  d.t[1],  d.L_hat[1], d.c_hat[1], d.p_hat[1], d.t_imp[1]]
    return lo, hi
