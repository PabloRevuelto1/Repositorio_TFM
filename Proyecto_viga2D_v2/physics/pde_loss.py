"""
physics/pde_loss.py
===================
Pérdida física de la elastodinámica 2D (ecuaciones de Navier-Cauchy en
TENSIÓN PLANA) para el modelo PINNsFormer paramétrico.

Todas las ecuaciones están escritas en variables ADIMENSIONALES
(ver config.py). Con  ξ = x/L,  η = y/c,  t̂ = t/T_REF  y los coeficientes
μ̂ = μ/E,  λ̂ = λ/E, la EDP queda:

  Residuo (interior):
    r_u = μ̂(U_ξξ/L̂² + U_ηη/ĉ²) + (λ̂+μ̂)(U_ξξ/L̂² + V_ξη/(L̂ĉ)) - U_t̂t̂
    r_v = μ̂(V_ξξ/L̂² + V_ηη/ĉ²) + (λ̂+μ̂)(U_ξη/(L̂ĉ) + V_ηη/ĉ²) - V_t̂t̂

  Tensiones adimensionales (σ̂ = σ/(E·U_REF/L_REF)):
    σ̂_xx = (λ̂+2μ̂) U_ξ/L̂ + λ̂ V_η/ĉ
    σ̂_yy = λ̂ U_ξ/L̂ + (λ̂+2μ̂) V_η/ĉ
    τ̂_xy = μ̂ (U_η/ĉ + V_ξ/L̂)

  Condiciones de contorno e iniciales:
    * Inicial (t̂=0):        U=V=0,  U_t̂=V_t̂=0
    * Empotramiento (ξ=0):  U=V=0
    * Superficies (η=±1):   σ̂_yy=0,  τ̂_xy=0
    * Extremo cargado (ξ=1): σ̂_xx=0,  τ̂_xy = τ̂_app(η, t̂)
        τ̂_app = -p̂ · (ĉ_REF/ĉ) · (1 - η²) · sigmoid(α(t̂ - t̂_imp))

La predicción "real" de la red es el primer token de la pseudo-secuencia
(índice 0); el residuo se evalúa sobre toda la secuencia para reforzar la
coherencia temporal, como en el paper.
"""

import torch


def _grad(outputs, inputs):
    """Derivada de `outputs` respecto de `inputs` (mismo shape que outputs)."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph=True, create_graph=True,
    )[0]


class ElastodynamicsLoss:
    """Calcula L_res, L_bc y L_ic para el problema de la viga empotrada 2D."""

    def __init__(self, model, material, dominio, ent):
        self.model = model
        self.mu = material.mu_hat          # μ̂ (orden unidad)
        self.lam = material.lam_hat        # λ̂ (orden unidad)
        self.dom = dominio
        self.alpha = dominio.alpha_ramp
        self.c_hat_ref = None              # se fija desde config en train/app
        self.w_res = ent.w_res
        self.w_load = ent.w_load
        self.w_bc = ent.w_bc
        self.w_ic = ent.w_ic

    # ------------------------------------------------------------------
    # Utilidades de evaluación del modelo
    # ------------------------------------------------------------------
    @staticmethod
    def _split(batch):
        """Separa un tensor (N,k,7) en hojas derivables (ξ,η,t) y parámetros."""
        xi = batch[:, :, 0:1].clone().requires_grad_(True)
        eta = batch[:, :, 1:2].clone().requires_grad_(True)
        t = batch[:, :, 2:3].clone().requires_grad_(True)
        params = batch[:, :, 3:7]          # (L̂, ĉ, p̂, t̂_imp), sin gradiente
        return xi, eta, t, params

    def _predict(self, xi, eta, t, params):
        out = self.model(xi, eta, t, params)
        return out[:, :, 0:1], out[:, :, 1:2]   # U, V

    def _first_derivs(self, U, V, xi, eta):
        """Derivadas primeras espaciales (para tensiones)."""
        U_xi = _grad(U, xi)
        U_eta = _grad(U, eta)
        V_xi = _grad(V, xi)
        V_eta = _grad(V, eta)
        return U_xi, U_eta, V_xi, V_eta

    def _stresses(self, U_xi, U_eta, V_xi, V_eta, L_hat, c_hat):
        """Tensiones adimensionales σ̂_xx, σ̂_yy, τ̂_xy."""
        u_x = U_xi / L_hat                  # ∂U/∂x adimensional
        v_y = V_eta / c_hat
        u_y = U_eta / c_hat
        v_x = V_xi / L_hat
        s_xx = (self.lam + 2 * self.mu) * u_x + self.lam * v_y
        s_yy = self.lam * u_x + (self.lam + 2 * self.mu) * v_y
        t_xy = self.mu * (u_y + v_x)
        return s_xx, s_yy, t_xy

    # ------------------------------------------------------------------
    # Componentes de la pérdida
    # ------------------------------------------------------------------
    def loss_residual(self, batch):
        """Residuo de Navier-Cauchy en el interior (toda la pseudo-secuencia)."""
        xi, eta, t, params = self._split(batch)
        L_hat, c_hat = params[:, :, 0:1], params[:, :, 1:2]
        U, V = self._predict(xi, eta, t, params)

        # Derivadas primeras
        U_xi, U_eta, V_xi, V_eta = self._first_derivs(U, V, xi, eta)
        # Derivadas segundas espaciales
        U_xixi = _grad(U_xi, xi)
        U_etaeta = _grad(U_eta, eta)
        V_xieta = _grad(V_xi, eta)
        V_xixi = _grad(V_xi, xi)
        V_etaeta = _grad(V_eta, eta)
        U_xieta = _grad(U_xi, eta)
        # Derivadas segundas temporales
        U_t = _grad(U, t)
        U_tt = _grad(U_t, t)
        V_t = _grad(V, t)
        V_tt = _grad(V_t, t)

        L2, c2, Lc = L_hat ** 2, c_hat ** 2, L_hat * c_hat
        mu, lam = self.mu, self.lam

        r_u = mu * (U_xixi / L2 + U_etaeta / c2) \
            + (lam + mu) * (U_xixi / L2 + V_xieta / Lc) - U_tt
        r_v = mu * (V_xixi / L2 + V_etaeta / c2) \
            + (lam + mu) * (U_xieta / Lc + V_etaeta / c2) - V_tt

        # Reescalado por ĉ² para acotar el MAL CONDICIONAMIENTO del residuo:
        # los términos de flexión U_ηη/ĉ² escalan como 1/ĉ²≈100 (ĉ~0.1) y
        # dominaban el residuo, de modo que el optimizador lo minimizaba
        # colapsando a la solución trivial en vez de atender a la carga. Al
        # multiplicar por ĉ² (por muestra) esos términos quedan O(1) y la
        # magnitud del residuo deja de aplastar al resto de pérdidas.
        r_u = r_u * c2
        r_v = r_v * c2

        return torch.mean(r_u ** 2) + torch.mean(r_v ** 2)

    def loss_clamp(self, batch):
        """Empotramiento ξ=0: U=V=0 (primer token)."""
        xi, eta, t, params = self._split(batch)
        U, V = self._predict(xi, eta, t, params)
        return torch.mean(U[:, 0] ** 2) + torch.mean(V[:, 0] ** 2)

    def loss_free_surface(self, batch):
        """Superficies libres η=±1: σ̂_yy=0, τ̂_xy=0."""
        xi, eta, t, params = self._split(batch)
        L_hat, c_hat = params[:, :, 0:1], params[:, :, 1:2]
        U, V = self._predict(xi, eta, t, params)
        U_xi, U_eta, V_xi, V_eta = self._first_derivs(U, V, xi, eta)
        _, s_yy, t_xy = self._stresses(U_xi, U_eta, V_xi, V_eta, L_hat, c_hat)
        return torch.mean(s_yy[:, 0] ** 2) + torch.mean(t_xy[:, 0] ** 2)

    def loss_load(self, batch):
        """Extremo cargado ξ=1: σ̂_xx=0 y τ̂_xy = τ̂_app(η, t̂)."""
        xi, eta, t, params = self._split(batch)
        L_hat, c_hat = params[:, :, 0:1], params[:, :, 1:2]
        p_hat, t_imp = params[:, :, 2:3], params[:, :, 3:4]
        U, V = self._predict(xi, eta, t, params)
        U_xi, U_eta, V_xi, V_eta = self._first_derivs(U, V, xi, eta)
        s_xx, _, t_xy = self._stresses(U_xi, U_eta, V_xi, V_eta, L_hat, c_hat)

        tau_app = self.applied_traction(eta, t, p_hat, c_hat, t_imp)
        return torch.mean(s_xx[:, 0] ** 2) + torch.mean((t_xy[:, 0] - tau_app[:, 0]) ** 2)

    def applied_traction(self, eta, t, p_hat, c_hat, t_imp):
        """
        Tracción cortante adimensional aplicada en el extremo libre (ξ=1):
            τ̂_app = -p̂ · (ĉ_REF/ĉ) · (1 - η²) · sigmoid(α(t̂ - t̂_imp))
        Perfil parabólico (evita singularidades) modulado por una rampa sigmoidea.
        """
        gate = torch.sigmoid(self.alpha * (t - t_imp))
        parabola = (1.0 - eta ** 2)
        return -p_hat * (self.c_hat_ref / c_hat) * parabola * gate

    def loss_initial(self, batch):
        """Condición inicial t̂=0: U=V=0 y U_t̂=V_t̂=0 (reposo absoluto)."""
        xi, eta, t, params = self._split(batch)
        U, V = self._predict(xi, eta, t, params)
        U_t = _grad(U, t)
        V_t = _grad(V, t)
        loss_disp = torch.mean(U[:, 0] ** 2) + torch.mean(V[:, 0] ** 2)
        loss_vel = torch.mean(U_t[:, 0] ** 2) + torch.mean(V_t[:, 0] ** 2)
        return loss_disp + loss_vel

    # ------------------------------------------------------------------
    # Pérdida total
    # ------------------------------------------------------------------
    def total(self, sets):
        """
        Pérdida total ponderada. `sets` es un dict de tensores con claves:
        res, clamp, load, top, bottom, ic.
        Devuelve (loss_total, dict_componentes).
        """
        l_res = self.loss_residual(sets["res"])
        l_load = self.loss_load(sets["load"])               # carga: peso propio
        l_bc = (self.loss_clamp(sets["clamp"])              # contorno homogéneo
                + self.loss_free_surface(sets["top"])
                + self.loss_free_surface(sets["bottom"]))
        l_ic = self.loss_initial(sets["ic"])

        loss = (self.w_res * l_res + self.w_load * l_load
                + self.w_bc * l_bc + self.w_ic * l_ic)
        return loss, {"res": l_res.item(), "load": l_load.item(),
                      "bc": l_bc.item(), "ic": l_ic.item()}
