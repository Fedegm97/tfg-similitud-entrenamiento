"""Figuras para la seccion 7 del paper. Formato IEEE, una columna."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

R = json.load(open("resultados.json"))

# Paleta categorica validada (slots 1 y 2 de la paleta de referencia)
AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": TINTA2,
    "axes.linewidth": 0.6,
    "xtick.color": TINTA2,
    "ytick.color": TINTA2,
    "text.color": TINTA,
    "axes.labelcolor": TINTA,
    "figure.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

ANCHO = 3.4   # ancho de columna IEEE, en pulgadas


def limpiar(ax):
    """Grid recesiva, sin marco superior/derecho."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)


# ============================================================
# FIGURA 1 — Compromiso recall / latencia (barrido de efSearch)
# ============================================================
# Una sola serie: sin leyenda, el titulo la nombra. Cada punto se
# etiqueta directamente con su efSearch (encoding secundario).
b = R["barrido"]
x = [p["ms"] for p in b]
y = [p["recall"] for p in b]
ef = [p["ef"] for p in b]

fig, ax = plt.subplots(figsize=(ANCHO, 2.3))
ax.plot(x, y, "-", color=AZUL, linewidth=1.6, zorder=2)
ax.plot(x, y, "o", color=AZUL, markersize=4.5,
        markeredgecolor="white", markeredgewidth=0.8, zorder=3)

desp = {8: (7, -4), 16: (7, -6), 32: (0, 7), 64: (0, 7),
        128: (0, 7), 256: (0, 7)}
for xi, yi, e in zip(x, y, ef):
    dx, dy = desp[e]
    ha = "center" if dy > 0 else "left"
    ax.annotate(f"{e}", (xi, yi), textcoords="offset points",
                xytext=(dx, dy), fontsize=7, color=TINTA2, ha=ha)

ax.set_xlabel("Latencia por consulta (ms)")
ax.set_ylabel("recall@10")
ax.set_ylim(0.9855, 1.0022)
ax.set_xlim(-0.0015, max(x) * 1.10)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
limpiar(ax)
fig.savefig("fig_barrido.pdf")
fig.savefig("fig_barrido.png")
plt.close(fig)
print("fig_barrido — ok")


# ============================================================
# FIGURA 2 — Escalado: latencia vs tamano del corpus
# ============================================================
# Dos series: leyenda presente + etiqueta directa sobre cada linea.
# Identidad por color + marcador + estilo de linea (legible en gris).
e = R["escalado"]
n = [p["n"] for p in e]
flat = [p["flat_ms"] for p in e]
hnsw = [p["hnsw_ms"] for p in e]

fig, ax = plt.subplots(figsize=(ANCHO, 2.3))
ax.plot(n, flat, "-o", color=NARANJA, linewidth=1.6, markersize=4.5,
        markeredgecolor="white", markeredgewidth=0.8,
        label="IndexFlatL2 (exacto)", zorder=3)
ax.plot(n, hnsw, "--s", color=AZUL, linewidth=1.6, markersize=4.5,
        markeredgecolor="white", markeredgewidth=0.8,
        label="IndexHNSWFlat", zorder=3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Vectores indexados")
ax.set_ylabel("Latencia por consulta (ms)")
ax.set_xticks(n)
ax.set_xticklabels([f"{v//1000}k" for v in n])
ax.minorticks_off()

# etiqueta directa junto a cada linea, dentro del area de dibujo
ax.annotate("exacto", (n[2], flat[2]), textcoords="offset points",
            xytext=(6, -12), fontsize=7.5, color=NARANJA, ha="left")
ax.annotate("HNSW", (n[2], hnsw[2]), textcoords="offset points",
            xytext=(6, 5), fontsize=7.5, color=AZUL, ha="left")

ax.set_ylim(top=flat[-1] * 3.2)
ax.legend(frameon=False, loc="upper left", handlelength=1.8,
          labelcolor=TINTA2, borderpad=0.2)
limpiar(ax)
fig.savefig("fig_escalado.pdf")
fig.savefig("fig_escalado.png")
plt.close(fig)
print("fig_escalado — ok")
