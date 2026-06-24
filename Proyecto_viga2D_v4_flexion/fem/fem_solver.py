"""
fem/fem_solver.py
=================
SOLUCIÓN DE REFERENCIA (*ground truth*) por el MÉTODO DE LOS ELEMENTOS FINITOS.

Resuelve EXACTAMENTE el mismo problema que la PINN —elastodinámica 2D en tensión
plana de una viga empotrada con una carga cortante en rampa en el extremo libre—
usando `scikit-fem`, y devuelve los campos en la MISMA malla y los MISMOS
instantes que `utils.inference.evaluate_fields`, de modo que ambos resultados son
directamente comparables nodo a nodo (sin interpolar).

Discretización:
  * Malla triangular estructurada cuyos vértices COINCIDEN con la rejilla regular
    (nx × ny) de la inferencia de la PINN  →  comparación punto a punto exacta.
  * Elemento vectorial P1 (desplazamientos (u, v) lineales por triángulo).
  * Constitutiva de TENSIÓN PLANA:  λ* = Eν/(1-ν²),  μ = E/(2(1+ν)).
  * Integración temporal de NEWMARK-β (β=1/4, γ=1/2): media de la aceleración,
    implícita e INCONDICIONALMENTE ESTABLE (no impone límite al paso de tiempo).
  * Carga: misma tracción cortante parabólica en η que la PINN, con el MISMO
    perfil temporal de rampa lineal  g(t) = min(t/t_ramp, 1).

Las tensiones (constantes por elemento en P1) se recuperan a nodos mediante una
PROYECCIÓN L2 (resolver una matriz de masa escalar), que suaviza el campo.

Todo en unidades FÍSICAS del SI: el resultado es comparable directamente con los
campos físicos que entrega la PINN.
"""

import numpy as np
from scipy.sparse.linalg import splu

from skfem import (MeshTri, Basis, FacetBasis, ElementVector, ElementTriP1,
                   BilinearForm, LinearForm, asm, enforce)
from skfem.helpers import dot, ddot, sym_grad, trace

import config as C


def _lame_plane_stress(material):
    """Parámetros de Lamé EFECTIVOS de tensión plana (reproducen σ=Dε en 2D)."""
    E, nu = material.E, material.nu
    lam = E * nu / (1.0 - nu ** 2)
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def _grid_index(mesh, nx, ny, L, c):
    """Índices (iy, ix) de la rejilla regular para cada nodo de la malla FEM."""
    dx = L / (nx - 1)
    dy = 2.0 * c / (ny - 1)
    ix = np.round(mesh.p[0] / dx).astype(int)
    iy = np.round((mesh.p[1] + c) / dy).astype(int)
    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    return iy, ix


def solve_fem(material=None, nx=45, ny=15, nt=40, substeps=None):
    """
    Resuelve la elastodinámica por FEM y devuelve un dict con la MISMA estructura
    que `utils.inference.evaluate_fields` (arrays (nt, ny, nx) en unidades físicas).
    """
    material = material or C.MATERIAL
    L, c, P_max = C.VIGA.L, C.VIGA.c, C.VIGA.P_max
    lam, mu = _lame_plane_stress(material)
    rho = material.rho

    # --- Malla: vértices = rejilla regular de la inferencia ---
    xs = np.linspace(0.0, L, nx)
    ys = np.linspace(-c, c, ny)
    mesh = MeshTri.init_tensor(xs, ys)
    e = ElementVector(ElementTriP1())
    basis = Basis(mesh, e)
    ndof = basis.N

    # --- Matrices de rigidez (tensión plana) y de masa ---
    @BilinearForm
    def stiffness(u, v, w):
        return (2.0 * mu * ddot(sym_grad(u), sym_grad(v))
                + lam * trace(sym_grad(u)) * trace(sym_grad(v)))

    @BilinearForm
    def mass(u, v, w):
        return rho * dot(u, v)

    K = asm(stiffness, basis)
    M = asm(mass, basis)

    # --- Vector de carga espacial (amplitud temporal unidad) ---
    # Tracción en ξ=1: t = (σ_xx, τ_xy)·n = (0, τ(y)) con τ parabólica.
    # Pico físico 3P_max/(4c) (integra a P_max sobre [-c,c]); signo negativo.
    right = mesh.facets_satisfying(lambda x: np.abs(x[0] - L) < 1e-9)
    fb = FacetBasis(mesh, e, facets=right)

    @LinearForm
    def load(v, w):
        y = w.x[1]
        tau = -(3.0 * P_max / (4.0 * c)) * (1.0 - (y / c) ** 2)
        return tau * v[1]

    F_spatial = asm(load, fb)

    # --- Empotramiento ξ=0: u=v=0 ---
    D = basis.get_dofs(lambda x: np.abs(x[0]) < 1e-9).flatten()

    # --- Parámetros de Newmark (media de la aceleración) ---
    substeps = substeps or C.VALIDACION.fem_substeps
    T_phys = C.DOMINIO.T_hat * material.T_ref
    t_ramp_phys = C.VIGA.t_ramp_hat * material.T_ref
    n_steps = (nt - 1) * substeps
    dt = T_phys / n_steps
    beta, gamma = 0.25, 0.5
    a0 = 1.0 / (beta * dt * dt)
    a1 = 1.0 / (beta * dt)
    a2 = 1.0 / (2.0 * beta) - 1.0
    a3 = dt * (1.0 - gamma)
    a4 = dt * gamma

    # Matriz efectiva constante → se factoriza UNA sola vez.
    Keff = enforce(K + a0 * M, D=D)
    lu = splu(Keff.tocsc())

    def gate(t):
        return min(t / t_ramp_phys, 1.0) if t_ramp_phys > 0 else 1.0

    # --- Bucle temporal ---
    u = np.zeros(ndof)
    vel = np.zeros(ndof)
    acc = np.zeros(ndof)            # reposo absoluto y carga nula en t=0 → acc0=0

    out_u = np.zeros((nt, ndof))
    out_u[0] = u
    out_idx = 1
    for step in range(1, n_steps + 1):
        t = step * dt
        rhs = gate(t) * F_spatial + M.dot(a0 * u + a1 * vel + a2 * acc)
        rhs[D] = 0.0                                  # empotramiento homogéneo
        u_new = lu.solve(rhs)
        acc_new = a0 * (u_new - u) - a1 * vel - a2 * acc
        vel_new = vel + a3 * acc + a4 * acc_new
        u, vel, acc = u_new, vel_new, acc_new
        if step % substeps == 0:
            out_u[out_idx] = u
            out_idx += 1

    # --- Recuperación de tensiones a nodos por proyección L2 ---
    sbasis = Basis(mesh, ElementTriP1())
    Ms = asm(BilinearForm(lambda p, q, w: p * q), sbasis)
    Ms_lu = splu(Ms.tocsc())

    def _proj(comp):
        """Proyección L2 a nodos de una componente de tensión (función de ∇u)."""
        @LinearForm
        def form(q, w):
            du = w["disp"].grad
            exx, eyy = du[0, 0], du[1, 1]
            exy = 0.5 * (du[0, 1] + du[1, 0])
            if comp == "xx":
                s = (lam + 2.0 * mu) * exx + lam * eyy
            elif comp == "yy":
                s = lam * exx + (lam + 2.0 * mu) * eyy
            else:                                    # "xy"
                s = 2.0 * mu * exy
            return s * q
        return form

    forms = {c_: _proj(c_) for c_ in ("xx", "yy", "xy")}

    # --- Mapeo nodos → rejilla (iy, ix) ---
    iy, ix = _grid_index(mesh, nx, ny, L, c)
    xdof = basis.nodal_dofs[0]      # dof de la componente x por nodo
    ydof = basis.nodal_dofs[1]

    def _to_grid_disp(vec, dof):
        g = np.zeros((ny, nx))
        g[iy, ix] = vec[dof]
        return g

    def _to_grid_nodal(nodal):
        g = np.zeros((ny, nx))
        g[iy, ix] = nodal
        return g

    u_f = np.zeros((nt, ny, nx))
    v_f = np.zeros((nt, ny, nx))
    sxx = np.zeros((nt, ny, nx))
    syy = np.zeros((nt, ny, nx))
    txy = np.zeros((nt, ny, nx))
    exx = np.zeros((nt, ny, nx))
    eyy = np.zeros((nt, ny, nx))
    exy = np.zeros((nt, ny, nx))

    eps_fac = 1.0 / material.E      # paso de tensión a deformación: ε = σ/E aprox.
    for i in range(nt):
        uvec = out_u[i]
        u_f[i] = _to_grid_disp(uvec, xdof)
        v_f[i] = _to_grid_disp(uvec, ydof)
        disp = basis.interpolate(uvec)
        sxx_n = Ms_lu.solve(asm(forms["xx"], sbasis, disp=disp))
        syy_n = Ms_lu.solve(asm(forms["yy"], sbasis, disp=disp))
        txy_n = Ms_lu.solve(asm(forms["xy"], sbasis, disp=disp))
        sxx[i] = _to_grid_nodal(sxx_n)
        syy[i] = _to_grid_nodal(syy_n)
        txy[i] = _to_grid_nodal(txy_n)

    von = np.sqrt(np.clip(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * txy ** 2, 0, None))
    # Deformaciones físicas a partir de las tensiones (Hooke, tensión plana):
    nu = material.nu
    exx = (sxx - nu * syy) / material.E
    eyy = (syy - nu * sxx) / material.E
    exy = txy / (2.0 * mu)

    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(-1.0, 1.0, ny)
    t_hat = np.linspace(0.0, C.DOMINIO.T_hat, nt)
    x_phys = (xi * L)[None, None, :] * np.ones((nt, ny, 1))
    y_phys = (eta * c)[None, :, None] * np.ones((nt, 1, nx))
    j0 = ny // 2

    return {
        "x": x_phys, "y": y_phys, "u": u_f, "v": v_f,
        "sxx": sxx, "syy": syy, "txy": txy, "von_mises": von,
        "exx": exx, "eyy": eyy, "exy": exy,
        "t_hat": t_hat, "t_phys": t_hat * material.T_ref,
        "tip_v": v_f[:, j0, -1], "xi": xi, "eta": eta, "j0": j0,
    }
