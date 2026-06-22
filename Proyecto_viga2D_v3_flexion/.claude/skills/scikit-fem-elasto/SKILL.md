---
name: scikit-fem-elasto
description: Patrones para el solver de Elementos Finitos de referencia (ground truth) en elastodinámica 2D con scikit-fem, usado en fem/fem_solver.py. Úsala al tocar el FEM, añadir recuperación de tensiones, cambiar el integrador temporal, o asegurar que el FEM y la PINN resuelven EXACTAMENTE el mismo problema. Cubre tensión plana, ensamblado, BCs, Newmark y proyección L2 de tensiones.
---

# FEM de referencia (scikit-fem) para elastodinámica 2D

Objetivo: resolver el **mismo** problema que la PINN como *ground truth* objetivo,
en la **misma malla e instantes**, para comparar campo a campo sin interpolar.

## Claves de equivalencia con la PINN (¡críticas!)
- **Malla = rejilla de inferencia**: `MeshTri.init_tensor(xs, ys)` con
  `xs=linspace(0,L,nx)`, `ys=linspace(-c,c,ny)` → los vértices coinciden nodo a nodo
  con la rejilla de la PINN. Comparación directa.
- **Mismo perfil de carga** `g(t̂)` (rampa lineal) y misma tracción parabólica.
- **Tensión plana** (igual que la PINN): usa Lamé efectivos
  `λ* = Eν/(1-ν²)`, `μ = E/(2(1+ν))`. Con ellos `σ = λ*·tr(ε)·I + 2μ·ε` reproduce
  exactamente la ley de tensión plana (verificable: σxx = E/(1-ν²)(εxx+ν·εyy)).

## Esqueleto del solver
```python
from skfem import (MeshTri, Basis, FacetBasis, ElementVector, ElementTriP1,
                   BilinearForm, LinearForm, asm, enforce)
from skfem.helpers import dot, ddot, sym_grad, trace
from scipy.sparse.linalg import splu

mesh = MeshTri.init_tensor(xs, ys)
e = ElementVector(ElementTriP1()); basis = Basis(mesh, e)

@BilinearForm
def stiffness(u, v, w):
    return 2*mu*ddot(sym_grad(u), sym_grad(v)) + lam*trace(sym_grad(u))*trace(sym_grad(v))
@BilinearForm
def mass(u, v, w):
    return rho*dot(u, v)
K, M = asm(stiffness, basis), asm(mass, basis)

# Carga: tracción (0, τ(y)) en facetas x=L; τ = -(3P/4c)(1-(y/c)²)
fb = FacetBasis(mesh, e, facets=mesh.facets_satisfying(lambda x: abs(x[0]-L) < 1e-9))
@LinearForm
def load(v, w):
    y = w.x[1]; tau = -(3*P_max/(4*c))*(1-(y/c)**2)
    return tau * v[1]          # componente y del vector test
F_spatial = asm(load, fb)

D = basis.get_dofs(lambda x: abs(x[0]) < 1e-9).flatten()   # empotramiento
```

## Integración temporal: Newmark-β (media de la aceleración)
- `β=1/4, γ=1/2`: implícito, **incondicionalmente estable** (sin límite de Δt).
- Matriz efectiva CONSTANTE → factorizar UNA vez: `Keff = enforce(K + a0·M, D=D)`,
  `lu = splu(Keff.tocsc())` con `a0 = 1/(β·Δt²)`.
- Por paso: `rhs = g(t)·F_spatial + M·(a0·u + a1·v + a2·a)`, `rhs[D]=0`, `u=lu.solve(rhs)`,
  luego actualizar aceleración y velocidad. IC: u=v=a=0 (carga nula en t=0).
- `substeps` entre instantes de salida para precisión (config `VALIDACION.fem_substeps`).

## Recuperación de tensiones (P1 → nodos)
P1 da deformación constante por elemento. Para tensiones nodales suaves, **proyección
L2**: resolver `Ms·s = b` con `Ms` masa escalar (`Basis(mesh, ElementTriP1())`) y
`b = asm(LinearForm(σ_comp·q), sbasis, disp=basis.interpolate(u_vec))`, accediendo a
`w["disp"].grad` (∂u_i/∂x_j) dentro del form. Una proyección por componente (xx,yy,xy).

## Salida
Devolver el **mismo dict** que `utils.inference.evaluate_fields` (claves x,y,u,v,
sxx,syy,txy,von_mises,exx,eyy,exy,t_hat,t_phys,tip_v,xi,eta,j0) para que `metrics` y
`app` lo traten igual. Verificación física: σ_VM máxima en el **empotramiento**.

## Instalación
`uv add scikit-fem` desde la raíz del repo (entorno uv). Es FEM puro en Python
(NumPy/SciPy), ligero, corre en local/WSL sin problema.
