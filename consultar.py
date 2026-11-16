"""
Consulta de un corredor externo al corpus
==========================================
TFG · Federico Garcia Monzon

Carga el indice FAISS ya construido y recupera los corredores mas
similares a un perfil que no esta indexado.

Tres formas de usarlo:

    python consultar.py
        Ejemplo por defecto: 30 semanas, volumen creciente de 28 a 52 km.

    python consultar.py --km 45 --ritmo 5.2 --semanas 30
        Genera un historial con esos valores. --km es el promedio semanal,
        --ritmo el ritmo tipico en min/km, --semanas cuantas entreno.
        Opcional: --progresion sube|estable|baja  (por defecto: sube)

    python consultar.py mis_semanas.csv
        Historial real. CSV con columnas 'distance' (km) y 'duration'
        (min), una fila por semana en orden cronologico.

Requiere en la misma carpeta: indice_hnsw.faiss, indice_params.npz
y perfiles.csv (los genera pipeline.py).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    sys.exit("Falta faiss. Instalalo con:  pip install faiss-cpu pandas numpy")

K = 5
REQUERIDOS = ["indice_hnsw.faiss", "indice_params.npz", "perfiles.csv"]


# ---------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------
ap = argparse.ArgumentParser(add_help=True)
ap.add_argument("csv", nargs="?", help="CSV con columnas distance y duration")
ap.add_argument("--km", type=float, help="promedio de km por semana")
ap.add_argument("--ritmo", type=float, default=5.5, help="min/km (default 5.5)")
ap.add_argument("--semanas", type=int, default=30, help="semanas entrenadas")
ap.add_argument("--progresion", choices=["sube", "estable", "baja"],
                default="sube", help="tendencia del volumen")
a = ap.parse_args()


# ---------------------------------------------------------------
# 1. Cargar indice y parametros
# ---------------------------------------------------------------
faltan = [f for f in REQUERIDOS if not os.path.exists(f)]
if faltan:
    sys.exit(f"Faltan archivos: {', '.join(faltan)}\n"
             f"Genera los con:  python pipeline.py run_ww_2019_w.csv")

ix = faiss.read_index("indice_hnsw.faiss")
p = np.load("indice_params.npz", allow_pickle=True)
mu, sigma, feats = p["mu"], p["sigma"], list(p["features"])
perfiles = pd.read_csv("perfiles.csv", index_col=0)

print(f"\nIndice: {ix.ntotal:,} corredores, {ix.d} dimensiones por perfil")


# ---------------------------------------------------------------
# 2. Construir el historial de consulta
# ---------------------------------------------------------------
def historial_sintetico(km, ritmo, semanas, tendencia):
    """
    Genera un historial con variacion semana a semana — un corredor real
    no repite el mismo numero — pero calibrado para que el promedio de
    distancia y la mediana de ritmo coincidan con lo pedido. Sin esa
    variacion, las features de desvio y percentil 90 darian cero.
    """
    rng = np.random.default_rng(7)
    if tendencia == "sube":
        base = np.linspace(km * 0.65, km * 1.35, semanas)
    elif tendencia == "baja":
        base = np.linspace(km * 1.35, km * 0.65, semanas)
    else:
        base = np.full(semanas, float(km))

    dist = (base + rng.normal(0, km * 0.10, semanas)).clip(km * 0.15)
    dist = np.round(dist * (km / dist.mean()), 1)          # media exacta

    r = rng.normal(ritmo, ritmo * 0.045, semanas)
    r = r + (ritmo - np.median(r))                          # mediana exacta
    dur = np.round(dist * r, 1)
    return pd.DataFrame({"distance": dist, "duration": dur})


if a.csv:
    semanas = pd.read_csv(a.csv)
    falta = {"distance", "duration"} - set(semanas.columns)
    if falta:
        sys.exit(f"Al CSV le faltan columnas: {', '.join(falta)}")
    print(f"Consulta: {a.csv} ({len(semanas)} semanas)")
else:
    km = a.km if a.km else 40.0
    semanas = historial_sintetico(km, a.ritmo, a.semanas, a.progresion)
    print(f"Consulta: {a.semanas} semanas, ~{km:.0f} km/sem, "
          f"ritmo {a.ritmo}, volumen {a.progresion}")


# ---------------------------------------------------------------
# 3. Vectorizar con la misma logica de la seccion 5
# ---------------------------------------------------------------
def vectorizar(s):
    s = s.copy()
    s["ritmo"] = s["duration"] / s["distance"].replace(0, np.nan)
    umbral = s["ritmo"].quantile(0.10)

    IF = (umbral / s["ritmo"]).clip(0, 1.5).fillna(0)
    s["tss"] = ((s["duration"] / 60.0) * IF**2 * 100).fillna(0)
    s["ctl"] = s["tss"].ewm(span=6, adjust=False).mean()
    s["tsb"] = s["ctl"] - s["tss"]

    act = s[s.distance > 0]
    v = {
        "dist_media"    : act["distance"].mean(),
        "dist_std"      : act["distance"].std(),
        "dist_p90"      : act["distance"].quantile(0.90),
        "dist_total"    : s["distance"].sum(),
        "ctl_final"     : s["ctl"].iloc[-1],
        "ctl_max"       : s["ctl"].max(),
        "tss_medio"     : act["tss"].mean(),
        "tsb_std"       : s["tsb"].std(),
        "ritmo_mediano" : act["ritmo"].median(),
        "ritmo_umbral"  : umbral,
        "semanas_act"   : len(act),
        "consistencia"  : len(act) / len(s),
        "progresion"    : (np.polyfit(np.arange(len(s)), s["ctl"], 1)[0]
                           if len(s) > 2 else 0.0),
    }
    x = np.nan_to_num(np.array([v[f] for f in feats], dtype=np.float64))

    # Clave: se normaliza con los estadisticos del CORPUS, no con los
    # del propio atleta. De otro modo el vector quedaria en otra escala
    # y las distancias no significarian nada.
    return ((x - mu) / (sigma + 1e-9)).astype(np.float32).reshape(1, -1), v


q, v = vectorizar(semanas)


# ---------------------------------------------------------------
# 4. Buscar
# ---------------------------------------------------------------
D, I = ix.search(q, K)

print(f"\nPerfil calculado")
print(f"  {v['dist_media']:.1f} km/sem  ·  ritmo {v['ritmo_mediano']:.2f}"
      f"  ·  CTL final {v['ctl_final']:.0f}"
      f"  ·  progresion {v['progresion']:+.2f}")

print(f"\n{K} corredores mas similares del corpus\n")
print(f"{'':>6} {'km/sem':>8} {'CTL':>8} {'ritmo':>7} {'sem':>5} "
      f"{'prog':>7} {'dist':>7}")
print(f"{'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*5} {'-'*7} {'-'*7}")
for rank, (idx, d) in enumerate(zip(I[0], D[0]), 1):
    r = perfiles.iloc[idx]
    print(f"{'#'+str(rank):>6} {r.dist_media:>8.1f} {r.ctl_final:>8.1f} "
          f"{r.ritmo_mediano:>7.2f} {r.semanas_act:>5.0f} "
          f"{r.progresion:>+7.2f} {d:>7.2f}")

vec = perfiles.iloc[I[0]]
print(f"\nRango de los vecinos")
print(f"  km/sem     {vec.dist_media.min():.1f} a {vec.dist_media.max():.1f}"
      f"   (consulta: {v['dist_media']:.1f})")
print(f"  ritmo      {vec.ritmo_mediano.min():.2f} a "
      f"{vec.ritmo_mediano.max():.2f}   (consulta: {v['ritmo_mediano']:.2f})")
print(f"  progresion {vec.progresion.min():+.2f} a {vec.progresion.max():+.2f}"
      f"   (consulta: {v['progresion']:+.2f})")
print(f"\n'dist' es la distancia euclidea en el espacio estandarizado: "
      f"mas chica, mas parecido.\n")
