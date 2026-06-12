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

    Entrada : tensores separados (xi, eta, t, params) para permitir la
              diferenciación automática respecto de las coordenadas.
                xi, eta, t  : (N, k, 1)  coordenadas adimensionales
                params      : (N, k, 4)  (L̂, ĉ, p̂, t̂_imp)
    Salida  : (N, k, 2) → (U, V) desplazamientos adimensionales.
    """

    def __init__(self, d_in, d_out, d_model, d_hidden, N, heads, in_lo=None, in_hi=None,
                 hard_bc=False, t_hat_max=1.0):
        super().__init__()

        # Restricciones DURAS de IC/empotramiento (ver config.ModeloCfg).
        self.hard_bc = hard_bc
        self.register_buffer("t_hat_max", torch.tensor(float(t_hat_max)))

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

    def forward(self, xi, eta, t, params):
        # Construye la secuencia de entrada y la normaliza a ~[-1, 1].
        src = torch.cat((xi, eta, t, params), dim=-1)          # (N, k, d_in)
        src = (src - self.in_center) / self.in_halfspan

        src = self.linear_emb(src)                             # (N, k, d_model)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)                     # (N, k, d_out)

        # Ansatz de restricción dura: U,V = ξ·(t̂/T_hat)² · Ñ(...). El factor se
        # anula en ξ=0 (empotramiento) y en t̂=0 (IC, con velocidad nula por el
        # cuadrado). La autodiferenciación atraviesa este factor, de modo que el
        # residuo de la EDP y las tensiones se calculan sobre la solución ya
        # restringida; train e inferencia quedan automáticamente consistentes.
        if self.hard_bc:
            tau = t / self.t_hat_max                           # (N, k, 1)
            factor = xi * tau * tau                            # 0 en ξ=0 y t̂=0
            output = output * factor
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
