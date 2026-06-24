"""
models/pinnsformer.py
=====================
Implementación de la arquitectura PINNsFormer (Zhao et al., ICLR 2024,
https://arxiv.org/abs/2307.11833) siguiendo FIELMENTE el repositorio
oficial de los autores (TFM/pinnsformer/model/pinnsformer.py):

    WaveAct → FeedForward → EncoderLayer / DecoderLayer → Encoder / Decoder
            → PINNsformer (embedding + encoder + decoder + cabeza MLP)

Componentes idénticos al original:
  * WaveAct:        activación de onda  f(x) = w1·sin(x) + w2·cos(x)
  * Encoder/Decoder con atención multi-cabeza (cross-attention en el decoder)
  * Esquema residual (x = x + Attn(x);  x = x + FF(x))

Extensiones para ESTE problema (elastodinámica 2D paramétrica):
  * Embedding de entrada de dimensión `d_in = 7`  →  (ξ, η, t̂, L̂, ĉ, p̂, t̂_imp)
    en lugar de 2, para soportar PINNs PARAMÉTRICAS (la red generaliza a
    distintas geometrías y cargas sin reentrenar).
  * Salida `d_out = 2`  →  (U, V) desplazamientos adimensionales.
  * Normalización de entrada a ~[-1, 1] mediante buffers (lo, hi) para
    mejorar el condicionamiento del entrenamiento.

El truco de la PSEUDO-SECUENCIA temporal (make_time_sequence) vive en
utils/sampling.py: cada punto se replica `k` veces desplazando t̂ un paso
pequeño, generando una secuencia que el transformer procesa para capturar
la dependencia temporal. La predicción "real" es el primer token (índice 0).
"""

import copy

import torch
import torch.nn as nn


def get_clones(module, N):
    """Crea N copias profundas e independientes de un módulo."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class WaveAct(nn.Module):
    """Activación de onda del paper: w1·sin(x) + w2·cos(x) con w1, w2 aprendibles."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class FeedForward(nn.Module):
    """Red feed-forward con activaciones de onda (idéntica al original)."""

    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    """Capa de codificador: self-attention + feed-forward con conexiones residuales."""

    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    """Capa de decodificador: cross-attention con la salida del encoder + feed-forward."""

    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs):
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    """Pila de N capas de codificador."""

    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.act(x)


class Decoder(nn.Module):
    """Pila de N capas de decodificador (cross-attention sobre la memoria del encoder)."""

    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x, e_outputs):
        for layer in self.layers:
            x = layer(x, e_outputs)
        return self.act(x)


class PINNsformer(nn.Module):
    """
    Modelo PINNsFormer paramétrico para elastodinámica 2D.

    Entrada : tensores separados (xi, eta, t) para permitir la diferenciación
              automática respecto de las coordenadas.
                xi, eta, t  : (N, k, 1)  coordenadas adimensionales
    Salida  : (N, k, 2) → (U, V) desplazamientos adimensionales.

    Escenario FIJO (proyecto simplificado): la geometría/carga es constante
    (config.VIGA), por lo que la red ya no recibe parámetros como entrada.
    """

    def __init__(self, d_in, d_out, d_model, d_hidden, N, heads, in_lo=None, in_hi=None,
                 hard_bc=False, ic_tau=1.0, hard_ic=False):
        super().__init__()

        # Restricciones DURAS de IC/empotramiento (ver config.ModeloCfg).
        # ic_tau = constante de subida de la envolvente temporal.
        # hard_ic: True → envolvente t̂²/(t̂²+τ²) (anula desplazamiento Y velocidad);
        #          False → envolvente t̂/(t̂+τ) (anula sólo desplazamiento; u̇(0)=0 queda
        #          como condición BLANDA → necesario para que emerja la oscilación).
        self.hard_bc = hard_bc
        self.hard_ic = hard_ic
        self.register_buffer("ic_tau", torch.tensor(float(ic_tau)))

        # Embedding inicial: proyecta las d_in entradas al espacio del transformer.
        self.linear_emb = nn.Linear(d_in, d_model)

        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)

        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out),
        )

        # Buffers de normalización de entrada a ~[-1, 1]: x_n = 2(x-lo)/(hi-lo) - 1
        if in_lo is None:
            in_lo = [0.0] * d_in
        if in_hi is None:
            in_hi = [1.0] * d_in
        lo = torch.tensor(in_lo, dtype=torch.float32)
        hi = torch.tensor(in_hi, dtype=torch.float32)
        self.register_buffer("in_center", (hi + lo) / 2.0)
        self.register_buffer("in_halfspan", (hi - lo) / 2.0)

    def forward(self, xi, eta, t):
        # Construye la secuencia de entrada y la normaliza a ~[-1, 1].
        src = torch.cat((xi, eta, t), dim=-1)                  # (N, k, d_in=3)
        src = (src - self.in_center) / self.in_halfspan

        src = self.linear_emb(src)                             # (N, k, d_model)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)                     # (N, k, d_out)

        # Ansatz de restricción dura: U,V = ξ · env(t̂) · Ñ(...). El factor ξ anula el
        # desplazamiento en el empotramiento (ξ=0); env(t̂) lo anula en t̂=0. Ambas
        # envolventes SATURAN a 1 para t̂≫τ (no crecen con el horizonte). Según hard_ic:
        #   - hard_ic=True : env = t̂²/(t̂²+τ²) ~ (t̂/τ)² en 0 → anula desplazamiento Y
        #                    velocidad (CI dura completa).
        #   - hard_ic=False: env = t̂/(t̂+τ) ~ t̂/τ en 0 → anula sólo el desplazamiento;
        #                    u̇(0)≠0 por construcción, así que la red debe imponer u̇(0)=0
        #                    como pérdida blanda → es lo que EXCITA la oscilación libre.
        # La autodiferenciación atraviesa el factor: residuo y tensiones se calculan sobre
        # la solución ya restringida; train e inferencia quedan automáticamente consistentes.
        if self.hard_bc:
            tau = self.ic_tau
            if self.hard_ic:
                t2 = t * t
                envelope = t2 / (t2 + tau * tau)               # ~ (t̂/τ)² en 0, → 1
            else:
                envelope = t / (t + tau)                       # ~ t̂/τ en 0, → 1
            output = output * (xi * envelope)                  # 0 en ξ=0 y t̂=0
        return output


def get_n_params(model):
    """Cuenta el número total de parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    """Inicialización Xavier para las capas lineales (como en el repo oficial)."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
