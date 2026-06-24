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

import config as C


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
        # Escenario FIJO (config.VIGA): geometría/carga constantes (ya no entran
        # a la red). El residuo y las tensiones usan estos adimensionales fijos.
        self.L_hat = C.VIGA.L_hat
        self.c_hat = C.VIGA.c_hat
        self.p_hat = C.VIGA.p_hat
        self.t_ramp = C.VIGA.t_ramp_hat       # duración de la rampa de carga
        self.tau_hat_peak = C.VIGA.tau_hat_peak   # pico adim. de la tracción aplicada
        # Equilibrio seccional (cortante/momento integral por sección)
        self.K_shear = C.VIGA.K_shear
        self.K_moment = C.VIGA.K_moment
        self.n_eta_eq = ent.n_eta_eq
        self.w_res = ent.w_res
        self.w_load = ent.w_load
        self.w_bc = ent.w_bc
        self.w_ic = ent.w_ic
        self.w_eq = ent.w_eq
        # Pesado CAUSAL del residuo (Wang et al. 2022): ver loss_residual.
        self.causal = ent.causal
        self.eps_causal = ent.eps_causal
        self.n_causal_bins = ent.n_causal_bins

    # ------------------------------------------------------------------
    # Utilidades de evaluación del modelo
    # ------------------------------------------------------------------
    @staticmethod
    def _split(batch):
        """Separa un tensor (N,k,3) en hojas derivables (ξ, η, t̂)."""
        xi = batch[:, :, 0:1].clone().requires_grad_(True)
        eta = batch[:, :, 1:2].clone().requires_grad_(True)
        t = batch[:, :, 2:3].clone().requires_grad_(True)
        return xi, eta, t

    def _predict(self, xi, eta, t):
        out = self.model(xi, eta, t)
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
        xi, eta, t = self._split(batch)
        U, V = self._predict(xi, eta, t)

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

        L_hat, c_hat = self.L_hat, self.c_hat
        L2, c2, Lc = L_hat ** 2, c_hat ** 2, L_hat * c_hat
        mu, lam = self.mu, self.lam

        r_u = mu * (U_xixi / L2 + U_etaeta / c2) \
            + (lam + mu) * (U_xixi / L2 + V_xieta / Lc) - U_tt
        r_v = mu * (V_xixi / L2 + V_etaeta / c2) \
            + (lam + mu) * (U_xieta / Lc + V_etaeta / c2) - V_tt

        # Reescalado de magnitud del residuo para EQUILIBRARLO con las demás
        # pérdidas (carga, contorno). Un factor constante NO cambia el cero del
        # residuo (la física), sólo su peso relativo en el gradiente.
        # Antes se usaba ĉ² (≈0.01), que hundía mean(r²) a ~1e-4 de su magnitud
        # natural → la EDP quedaba INFRA-impuesta frente al contorno (en el régimen
        # de flexión la EDP es la que genera la forma flexada, no se puede debilitar).
        # Con ĉ (≈0.1) el residuo queda del mismo orden que load/bc. (El factor NO
        # rebalancea los términos INTERNOS U_ηη/ĉ² vs U_t̂t̂ —eso es intrínseco—; sólo
        # fija la magnitud global; el ajuste fino se hace con w_res.) Ver docs/05 §4.
        r_u = r_u * c_hat
        r_v = r_v * c_hat

        # Residuo cuadrático POR PUNTO (media sobre la pseudo-secuencia k).
        rp = (r_u ** 2 + r_v ** 2).mean(dim=1).squeeze(-1)   # (N,)
        if not self.causal:
            return rp.mean()                                 # = mean(r_u²)+mean(r_v²)
        return self._causal_residual(rp, t)

    def _causal_residual(self, rp, t):
        """
        Pesado CAUSAL del residuo (Wang et al., "Respecting causality is all you
        need for training PINNs", 2022) — rompe la degeneración cuasi-estática↔dinámica.

        El problema (docs/05 §4-septies): la solución cuasi-estática y la dinámica tienen
        residuo EDP casi idéntico (difieren en ~ω̂²≈0.018, bajo el suelo de convergencia),
        así que una pérdida GLOBAL en el tiempo no las distingue y elige la cuasi-estática
        (que además viola la CI de reposo, u̇(0)≠0). El pesado causal obliga a satisfacer
        el residuo en ORDEN TEMPORAL: cada franja t̂ sólo "pesa" cuando las anteriores ya
        están resueltas, w_i = exp(-ε·Σ_{j<i} L_j) (con w detenido del grafo). Así la red
        propaga el reposo inicial hacia adelante —como un FEM que marcha— y la oscilación
        deja de ser invisible para el optimizador.
        """
        t0 = t[:, 0, 0].detach()                             # tiempo base por punto (N,)
        Nt = self.n_causal_bins
        tmin, tmax = t0.min(), t0.max()
        idx = ((t0 - tmin) / (tmax - tmin + 1e-12) * Nt).long().clamp(0, Nt - 1)

        # Residuo medio por franja temporal (conserva gradiente vía rp).
        Lsum = rp.new_zeros(Nt).index_add(0, idx, rp)
        cnt = rp.new_zeros(Nt).index_add(0, idx, torch.ones_like(rp))
        Lbin = Lsum / cnt.clamp(min=1.0)                     # (Nt,)

        # Peso causal: exponencial del residuo ACUMULADO de las franjas anteriores.
        Cprev = torch.cumsum(Lbin, 0) - Lbin                 # Σ_{j<i} L_j
        w = torch.exp(-self.eps_causal * Cprev).detach()
        mask = (cnt > 0).float()                             # ignora franjas vacías
        return (w * mask * Lbin).sum() / (w * mask).sum().clamp(min=1e-12)

    def loss_clamp(self, batch):
        """Empotramiento ξ=0: U=V=0 (primer token)."""
        xi, eta, t = self._split(batch)
        U, V = self._predict(xi, eta, t)
        return torch.mean(U[:, 0] ** 2) + torch.mean(V[:, 0] ** 2)

    def loss_free_surface(self, batch):
        """Superficies libres η=±1: σ̂_yy=0, τ̂_xy=0."""
        xi, eta, t = self._split(batch)
        U, V = self._predict(xi, eta, t)
        U_xi, U_eta, V_xi, V_eta = self._first_derivs(U, V, xi, eta)
        _, s_yy, t_xy = self._stresses(U_xi, U_eta, V_xi, V_eta, self.L_hat, self.c_hat)
        return torch.mean(s_yy[:, 0] ** 2) + torch.mean(t_xy[:, 0] ** 2)

    def loss_load(self, batch):
        """Extremo cargado ξ=1: σ̂_xx=0 y τ̂_xy = τ̂_app(η, t̂)."""
        xi, eta, t = self._split(batch)
        U, V = self._predict(xi, eta, t)
        U_xi, U_eta, V_xi, V_eta = self._first_derivs(U, V, xi, eta)
        s_xx, _, t_xy = self._stresses(U_xi, U_eta, V_xi, V_eta, self.L_hat, self.c_hat)

        tau_app = self.applied_traction(eta, t)
        return torch.mean(s_xx[:, 0] ** 2) + torch.mean((t_xy[:, 0] - tau_app[:, 0]) ** 2)

    def applied_traction(self, eta, t):
        """
        Tracción cortante adimensional aplicada en el extremo libre (ξ=1):
            τ̂_app = -τ̂_peak · (1 - η²) · g(t̂)
        con τ̂_peak = (3·P_max/4c)/SIGMA_SCALE (ver config.Viga.tau_hat_peak; ~0.015
        en el régimen de flexión) y RAMPA LINEAL g(t̂)=min(t̂/t_ramp,1).
        Perfil parabólico en η (evita singularidades en las esquinas).
        """
        gate = torch.clamp(t / self.t_ramp, 0.0, 1.0)
        parabola = (1.0 - eta ** 2)
        return -self.tau_hat_peak * parabola * gate

    def loss_equilibrium(self, batch):
        """
        EQUILIBRIO SECCIONAL — constraint GLOBAL que rompe el colapso trivial.

        El problema es de flexión CONTROLADA POR FUERZA y la EDP del interior es
        homogénea (ρü=∇·σ): u≈0 la satisface con residuo ≈0, así que la red colapsa
        a una solución casi nula (la flexión global no la "ve" el residuo local). El
        equilibrio integral SÍ la fija: en toda sección ξ, la resultante de cortante
        y el momento flector valen (consecuencia exacta de ∇·σ=0, NO datos):

            V̂(ξ,t̂) = ∫_{-1}^{1} τ̂_xy dη          = -K_shear · g(t̂)
            M̂(ξ,t̂) = ∫_{-1}^{1} σ̂_xx · η dη       = +K_moment · (1-ξ) · g(t̂)

        con g(t̂)=min(t̂/t_ramp,1) (nivel cuasi-estático de la carga). Se impone BLANDO:
        fija el esqueleto de amplitud; la desviación dinámica (inercia) la aporta el
        residuo. `batch` viene AGRUPADO por secciones de n_eta_eq nodos en η.
        """
        xi, eta, t = self._split(batch)
        U, V = self._predict(xi, eta, t)
        U_xi, U_eta, V_xi, V_eta = self._first_derivs(U, V, xi, eta)
        s_xx, _, t_xy = self._stresses(U_xi, U_eta, V_xi, V_eta, self.L_hat, self.c_hat)

        # Reagrupa por sección: (n_sec, n_eta). Se usa el primer token (índice 0).
        ne = self.n_eta_eq
        sxx = s_xx[:, 0, 0].reshape(-1, ne)
        txy = t_xy[:, 0, 0].reshape(-1, ne)
        eta_s = eta[:, 0, 0].reshape(-1, ne)           # rejilla η por sección
        xi_s = xi[:, 0, 0].reshape(-1, ne)[:, 0]       # ξ de cada sección
        t_s = t[:, 0, 0].reshape(-1, ne)[:, 0]         # t̂ de cada sección

        # Integral por cuadratura trapezoidal sobre η (rejilla uniforme [-1,1]).
        deta = 2.0 / (ne - 1)

        def _trapz(f):                                  # ∫ f dη  por sección → (n_sec,)
            return deta * (f.sum(dim=1) - 0.5 * (f[:, 0] + f[:, -1]))

        M = _trapz(sxx * eta_s)                          # momento flector adim.
        Vsh = _trapz(txy)                                # cortante resultante adim.

        gate = torch.clamp(t_s / self.t_ramp, 0.0, 1.0)
        target_M = self.K_moment * (1.0 - xi_s) * gate
        target_V = -self.K_shear * gate

        # El equilibrio se impone PLENO en TODA la ventana temporal. Es física EXACTA a
        # todo instante: la FORMA del momento M̂∝(1-ξ) (máximo en el empotramiento) y del
        # cortante V̂=cte son consecuencia de ∇·σ=0 SIEMPRE, no sólo durante la rampa.
        # (Histórico: una relajación exp(-(t̂-t_ramp)/τ) apagaba este término tras la rampa
        # para "liberar la oscilación"; fue un ERROR — apagaba el ÚNICO término que fija la
        # distribución de tensión, y la σ_VM máx. migraba del empotramiento al centro y caía
        # a ~40 %. El rebote NO sale de soltar el equilibrio; ver docs/05 §4-sexies.)
        return (torch.mean((M - target_M) ** 2)
                + torch.mean((Vsh - target_V) ** 2))

    def loss_initial(self, batch):
        """Condición inicial t̂=0: U=V=0 y U_t̂=V_t̂=0 (reposo absoluto)."""
        xi, eta, t = self._split(batch)
        U, V = self._predict(xi, eta, t)
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
        l_eq = self.loss_equilibrium(sets["eq"])            # equilibrio seccional

        loss = (self.w_res * l_res + self.w_load * l_load
                + self.w_bc * l_bc + self.w_ic * l_ic + self.w_eq * l_eq)
        return loss, {"res": l_res.item(), "load": l_load.item(),
                      "bc": l_bc.item(), "ic": l_ic.item(), "eq": l_eq.item()}
