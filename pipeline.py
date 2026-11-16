"""
Pipeline PoC — perfiles de corredores indexados en FAISS
=========================================================
TFG · Federico Garcia Monzon

  CSV semanal -> perfil por atleta -> vector -> FAISS (HNSW) -> evaluacion

Uso:
    python pipeline.py run_ww_2019_w.csv
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import faiss

RUTA      = sys.argv[1] if len(sys.argv) > 1 else "run_ww_2019_w.csv"
MIN_SEM   = 20        # semanas activas minimas
K         = 10        # vecinos a recuperar
N_QUERIES = 1000      # consultas para la evaluacion
N_REPS    = 5         # repeticiones para medir latencia
SEED      = 42

# HNSW
M                = 32
EF_CONSTRUCTION  = 80
EF_SEARCH        = 64

rng = np.random.default_rng(SEED)
OUT = {}   # resultados para volcar a JSON

print("=" * 68)
print("PIPELINE PoC — FAISS + HNSW sobre perfiles de corredores")
print("=" * 68)


# ================================================================
# 1. CARGA Y FILTRADO
# ================================================================
df = pd.read_csv(RUTA, usecols=["datetime", "athlete", "distance",
                                "duration", "gender", "age_group",
                                "country", "major"])
df["datetime"] = pd.to_datetime(df["datetime"])
anio = int(df["datetime"].dt.year.mode()[0])

atletas_total = int(df.athlete.nunique())

print(f"\n[1] Carga")
print(f"    Registros semanales : {len(df):,}")
print(f"    Atletas             : {atletas_total:,}")
print(f"    Anio                : {anio}")

meta = df.drop_duplicates("athlete").set_index("athlete")
con_major = meta.index[meta["major"].astype(str).str.contains(str(anio))]

activas = df[df.distance > 0].groupby("athlete").size()
suficientes = activas.index[activas >= MIN_SEM]

antes = len(df)
ritmo = df["duration"] / df["distance"].replace(0, np.nan)
df = df[(df.distance <= 250) & (ritmo.isna() | ritmo.between(2.5, 15))]
desc = antes - len(df)

sel = con_major.intersection(suficientes)
df = df[df.athlete.isin(sel)]

print(f"    Outliers descartados        : {desc:,} ({desc/antes*100:.2f}%)")
print(f"    Con maraton {anio}            : {len(con_major):,}")
print(f"    Con >= {MIN_SEM} semanas activas : {len(suficientes):,}")
print(f"    >>> CORPUS FINAL            : {len(sel):,} atletas")

OUT["corpus"] = {
    "registros": int(antes), "atletas_total": atletas_total,
    "con_major": int(len(con_major)), "suficientes": int(len(suficientes)),
    "outliers": int(desc), "final": int(len(sel)),
}


# ================================================================
# 1b. CARACTERIZACION DEL CORPUS
# ================================================================
# Cifras descriptivas que el paper reporta en la seccion del conjunto
# de datos. Se calculan aca, sobre el corpus ya seleccionado, para que
# ninguna quede transcrita a mano en el texto. Ojo con el denominador:
# son los 14.612 atletas retenidos, no los 36.412 del archivo.
mm = meta.reindex(sorted(sel))
n = len(mm)

# Distancia semanal, solo en semanas con actividad.
act_km = df.loc[df.distance > 0, "distance"]

# Maraton objetivo. Un atleta puede declarar mas de un destino en el
# mismo anio; el extract toma el primero, de modo que cada atleta
# cuenta una sola vez y los conteos suman el total del corpus.
ciudad_sel = (mm["major"].astype(str)
              .str.extract(rf"([A-Z ]+) {anio}")[0].str.strip())
por_maraton = ciudad_sel.value_counts()

OUT["descriptivo"] = {
    "n": int(n),
    "dist_sem_mediana": float(act_km.median()),
    "dist_sem_p95": float(act_km.quantile(0.95)),
    "registros_retenidos": int(len(df)),
    "registros_52_semanas": int(n * 52),
    "registros_faltantes": int(n * 52 - len(df)),
    "genero": {str(k): float(v / n * 100)
               for k, v in mm["gender"].value_counts().items()},
    "grupo_etario": {str(k): float(v / n * 100)
                     for k, v in mm["age_group"].value_counts().items()},
    "paises": int(mm["country"].nunique()),
    "por_maraton": {str(k): {"atletas": int(v), "pct": float(v / n * 100)}
                    for k, v in por_maraton.items()},
    "dos_mas_frecuentes_pct": float(por_maraton.iloc[:2].sum() / n * 100),
}

print(f"\n[1b] Caracterizacion del corpus")
print(f"    Distancia semanal   : mediana {act_km.median():.1f} km, "
      f"p95 {act_km.quantile(0.95):.1f} km")
print(f"    Registros retenidos : {len(df):,} de {n*52:,} posibles "
      f"({(n*52-len(df))/(n*52)*100:.2f}% faltantes)")
print(f"    Genero              : " + ", ".join(
    f"{k} {v/n*100:.1f}%" for k, v in mm["gender"].value_counts().items()))
print(f"    Grupo etario        : " + ", ".join(
    f"{k} {v/n*100:.1f}%" for k, v in mm["age_group"].value_counts().items()))
print(f"    Paises              : {mm['country'].nunique()}")
print(f"    {'maraton':<12} | {'atletas':>7} | {'%':>5}")
print(f"    {'-'*12}-+-{'-'*7}-+-{'-'*5}")
for k, v in por_maraton.items():
    print(f"    {k:<12} | {v:>7,} | {v/n*100:>4.1f}")
print(f"    {'TOTAL':<12} | {por_maraton.sum():>7,} | "
      f"{por_maraton.sum()/n*100:>4.1f}")


# ================================================================
# 2. METRICAS DE CARGA
# ================================================================
# Datos semanales, no diarios. Se adaptan las ventanas del modelo
# TRIMP: CTL = EWMA de 6 semanas (~42 d), ATL = semana actual
# (la ventana de 7 d coincide con la resolucion del dato).
df = df.sort_values(["athlete", "datetime"])

df["ritmo"] = df["duration"] / df["distance"].replace(0, np.nan)
umbral = df.groupby("athlete")["ritmo"].quantile(0.10).rename("umbral")
df = df.join(umbral, on="athlete")

IF = (df["umbral"] / df["ritmo"]).clip(0, 1.5).fillna(0)
horas = df["duration"] / 60.0
df["tss"] = (horas * IF**2 * 100).fillna(0)

df["ctl"] = (df.groupby("athlete")["tss"]
               .transform(lambda s: s.ewm(span=6, adjust=False).mean()))
df["atl"] = df["tss"]
df["tsb"] = df["ctl"] - df["atl"]

print(f"\n[2] Metricas de carga (TSS, CTL, ATL, TSB)")
print(f"    TSS  media {df.tss.mean():6.1f}  mediana {df.tss.median():6.1f}")
print(f"    CTL  media {df.ctl.mean():6.1f}  mediana {df.ctl.median():6.1f}")

OUT["carga"] = {
    "tss_media": float(df.tss.mean()), "tss_mediana": float(df.tss.median()),
    "ctl_media": float(df.ctl.mean()), "ctl_mediana": float(df.ctl.median()),
}


# ================================================================
# 3. VECTOR DE PERFIL
# ================================================================
g = df.groupby("athlete")
act = df[df.distance > 0].groupby("athlete")

perfil = pd.DataFrame({
    # volumen
    "dist_media"    : act["distance"].mean(),
    "dist_std"      : act["distance"].std(),
    "dist_p90"      : act["distance"].quantile(0.90),
    "dist_total"    : g["distance"].sum(),
    # carga
    "ctl_final"     : g["ctl"].last(),
    "ctl_max"       : g["ctl"].max(),
    "tss_medio"     : act["tss"].mean(),
    "tsb_std"       : g["tsb"].std(),
    # ritmo
    "ritmo_mediano" : act["ritmo"].median(),
    "ritmo_umbral"  : g["umbral"].first(),
    # dinamica
    "semanas_act"   : act.size(),
    "consistencia"  : act.size() / g.size(),
    "progresion"    : g["ctl"].apply(
                        lambda s: np.polyfit(np.arange(len(s)), s, 1)[0]
                                  if len(s) > 2 else 0.0),
}).fillna(0)

# Las variables demograficas (genero, grupo etario) y la maraton
# objetivo NO se incorporan al vector: se reservan como atributos
# externos para la validacion semantica. Si formaran parte de la
# representacion, comprobar que los vecinos las comparten seria
# una verificacion circular.
m = df.drop_duplicates("athlete").set_index("athlete")

perfil = perfil.loc[sorted(sel)]
DIM = perfil.shape[1]

print(f"\n[3] Vectores de perfil")
print(f"    Dimension : {DIM}")
print(f"    Features  : {', '.join(perfil.columns)}")

mu, sigma = perfil.values.mean(0), perfil.values.std(0)
X = ((perfil.values - mu) / (sigma + 1e-9)).astype(np.float32)
print(f"    Matriz    : {X.shape}  ({X.nbytes/1e6:.2f} MB)")

OUT["vector"] = {"dim": int(DIM), "n": int(len(X)),
                 "features": list(perfil.columns),
                 "mb": float(X.nbytes / 1e6)}


# ================================================================
# 4. INDICES FAISS
# ================================================================
def construir_exacto(Xd):
    t0 = time.perf_counter()
    ix = faiss.IndexFlatL2(Xd.shape[1])
    ix.add(Xd)
    return ix, time.perf_counter() - t0

def construir_hnsw(Xd, m=M, efc=EF_CONSTRUCTION, efs=EF_SEARCH):
    t0 = time.perf_counter()
    ix = faiss.IndexHNSWFlat(Xd.shape[1], m)
    ix.hnsw.efConstruction = efc
    ix.add(Xd)
    t = time.perf_counter() - t0
    ix.hnsw.efSearch = efs
    return ix, t

ix_exacto, t_ex = construir_exacto(X)
ix_hnsw,   t_hn = construir_hnsw(X)

mem_flat = X.nbytes
mem_hnsw = X.nbytes + len(X) * M * 2 * 4   # aprox: vectores + aristas

print(f"\n[4] Construccion de indices")
print(f"    IndexFlatL2    : {t_ex*1000:9.1f} ms")
print(f"    IndexHNSWFlat  : {t_hn*1000:9.1f} ms   (M={M}, efC={EF_CONSTRUCTION})")
print(f"    Ratio          : {t_hn/t_ex:9.0f}x")

OUT["construccion"] = {"flat_ms": t_ex*1000, "hnsw_ms": t_hn*1000,
                       "ratio": t_hn/t_ex,
                       "mem_flat_mb": mem_flat/1e6, "mem_hnsw_mb": mem_hnsw/1e6}


# ================================================================
# 5. RECALL Y LATENCIA
# ================================================================
qi = rng.choice(len(X), size=min(N_QUERIES, len(X)), replace=False)
Q = X[qi]

def sin_self(I, q_idx):
    """Descarta el propio atleta del resultado y deja K vecinos."""
    out = np.empty((len(I), K), dtype=I.dtype)
    for i, (fila, q) in enumerate(zip(I, q_idx)):
        v = fila[fila != q][:K]
        if len(v) < K:
            v = np.concatenate([v, fila[:K - len(v)]])[:K]
        out[i] = v
    return out

def medir(ix, reps=N_REPS):
    """Devuelve (indices, mejor tiempo total) sobre varias repeticiones."""
    mejor, I = None, None
    for _ in range(reps):
        t0 = time.perf_counter()
        _, R = ix.search(Q, K + 1)
        dt = time.perf_counter() - t0
        if mejor is None or dt < mejor:
            mejor, I = dt, R
    return I, mejor

R_ex, t_qex = medir(ix_exacto)
R_hn, t_qhn = medir(ix_hnsw)
I_ex, I_hn = sin_self(R_ex, qi), sin_self(R_hn, qi)

recall = sum(len(set(a) & set(b)) for a, b in zip(I_ex, I_hn)) / (len(Q) * K)

print(f"\n[5] Busqueda ({len(Q)} consultas, k={K}, mejor de {N_REPS})")
print(f"    Exacto : {t_qex*1000:7.2f} ms total | {t_qex/len(Q)*1000:.4f} ms/consulta")
print(f"    HNSW   : {t_qhn*1000:7.2f} ms total | {t_qhn/len(Q)*1000:.4f} ms/consulta")
print(f"    Speedup: {t_qex/t_qhn:.2f}x")
print(f"\n    >>> recall@{K} = {recall:.4f}")

OUT["busqueda"] = {
    "n_queries": int(len(Q)), "k": K,
    "flat_ms_total": t_qex*1000, "hnsw_ms_total": t_qhn*1000,
    "flat_ms_query": t_qex/len(Q)*1000, "hnsw_ms_query": t_qhn/len(Q)*1000,
    "speedup": t_qex/t_qhn, "recall": recall,
}

# Punto de equilibrio: a partir de cuantas consultas el sobrecosto de
# construir el grafo se amortiza con lo que ahorra por consulta.
# Se deriva de las mediciones de arriba, no se calcula a mano.
dc = OUT["construccion"]["hnsw_ms"] - OUT["construccion"]["flat_ms"]
dq = OUT["busqueda"]["flat_ms_query"] - OUT["busqueda"]["hnsw_ms_query"]
equilibrio = dc / dq
OUT["busqueda"]["equilibrio_consultas"] = float(equilibrio)
print(f"    Punto de equilibrio: {equilibrio:,.0f} consultas "
      f"({dc:.1f} ms de construccion / {dq:.4f} ms ahorrados por consulta)")


# ================================================================
# 6. PROXY SEMANTICO
# ================================================================
ids = perfil.index.to_numpy()
mm  = m.reindex(ids)
ciudad = (mm["major"].astype(str)
          .str.extract(rf"([A-Z ]+) {anio}")[0].str.strip().to_numpy())
gen  = mm["gender"].to_numpy()
edad = mm["age_group"].to_numpy()

def tasa(attr, I):
    return float(np.mean([np.mean(attr[v] == attr[q]) for q, v in zip(qi, I)]))

def azar(attr):
    p = pd.Series(attr).value_counts(normalize=True)
    return float((p**2).sum())

print(f"\n[6] Proxy semantico — atributos NO incluidos en el vector")
print(f"    {'atributo':<14} | {'observado':>10} | {'azar':>7} | {'lift':>6}")
print(f"    {'-'*14}-+-{'-'*10}-+-{'-'*7}-+-{'-'*6}")
OUT["proxy"] = {}
for nom, attr in [("maraton", ciudad), ("genero", gen), ("grupo etario", edad)]:
    o, a = tasa(attr, I_hn), azar(attr)
    print(f"    {nom:<14} | {o:>10.4f} | {a:>7.4f} | {o/a:>5.2f}x")
    OUT["proxy"][nom] = {"observado": o, "azar": a, "lift": o/a}


# ================================================================
# 7. BARRIDO DE efSearch
# ================================================================
print(f"\n[7] Compromiso recall / latencia")
print(f"    {'efSearch':>9} | {'recall@10':>10} | {'ms/consulta':>12}")
print(f"    {'-'*9}-+-{'-'*10}-+-{'-'*12}")
OUT["barrido"] = []
for ef in [8, 16, 32, 64, 128, 256]:
    ix_hnsw.hnsw.efSearch = ef
    mejor = None
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        _, R = ix_hnsw.search(Q, K + 1)
        dt = time.perf_counter() - t0
        mejor = dt if mejor is None else min(mejor, dt)
    I = sin_self(R, qi)
    rc = sum(len(set(a) & set(b)) for a, b in zip(I_ex, I)) / (len(Q) * K)
    ms = mejor / len(Q) * 1000
    print(f"    {ef:>9} | {rc:>10.4f} | {ms:>12.4f}")
    OUT["barrido"].append({"ef": ef, "recall": rc, "ms": ms})
ix_hnsw.hnsw.efSearch = EF_SEARCH


# ================================================================
# 8. ESCALADO — como se comporta el indice al crecer el corpus
# ================================================================
# El corpus real (14.612 x 13) queda por debajo del regimen donde
# HNSW se vuelve indispensable. Para situar ese punto se replica el
# corpus con ruido gaussiano leve y se repite la medicion.
print(f"\n[8] Escalado (corpus replicado con ruido)")
print(f"    {'N':>10} | {'flat ms':>9} | {'hnsw ms':>9} | {'speedup':>8} | {'recall':>7}")
print(f"    {'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")
OUT["escalado"] = []
rng_e = np.random.default_rng(SEED)
NQ_E = 300
for factor in [1, 4, 16, 64]:
    if factor == 1:
        Xs = X
    else:
        reps = [X] + [X + rng_e.normal(0, 0.05, X.shape).astype(np.float32)
                      for _ in range(factor - 1)]
        Xs = np.vstack(reps).astype(np.float32)

    ixe, _ = construir_exacto(Xs)
    ixh, _ = construir_hnsw(Xs)
    qs = rng_e.choice(len(Xs), size=NQ_E, replace=False)
    Qs = Xs[qs]

    def medir_s(ix):
        mejor, Rr = None, None
        for _ in range(3):
            t0 = time.perf_counter()
            _, R = ix.search(Qs, K + 1)
            dt = time.perf_counter() - t0
            if mejor is None or dt < mejor:
                mejor, Rr = dt, R
        return Rr, mejor

    Re, te = medir_s(ixe)
    Rh, th = medir_s(ixh)
    Ie = np.array([f[f != q][:K] for f, q in zip(Re, qs)])
    Ih = np.array([f[f != q][:K] for f, q in zip(Rh, qs)])
    rc = sum(len(set(a) & set(b)) for a, b in zip(Ie, Ih)) / (NQ_E * K)

    fms, hms = te/NQ_E*1000, th/NQ_E*1000
    print(f"    {len(Xs):>10,} | {fms:>9.4f} | {hms:>9.4f} | "
          f"{fms/hms:>7.1f}x | {rc:>7.4f}")
    OUT["escalado"].append({"n": int(len(Xs)), "flat_ms": fms,
                            "hnsw_ms": hms, "speedup": fms/hms, "recall": rc})
    del ixe, ixh
    if factor > 1:
        del Xs


# ================================================================
# 9. INSPECCION CUALITATIVA
# ================================================================
# Se eligen perfiles de percentiles opuestos de volumen para verificar
# que los vecinos son coherentes desde el conocimiento del dominio.
print(f"\n[9] Inspeccion cualitativa")

vol = perfil["dist_media"].to_numpy()
ctlf = perfil["ctl_final"].to_numpy()
prog = perfil["progresion"].to_numpy()

casos = {
    "volumen bajo"      : int(np.argsort(vol)[int(0.05 * len(vol))]),
    "volumen alto"      : int(np.argsort(vol)[int(0.95 * len(vol))]),
    "progresion alta"   : int(np.argsort(prog)[int(0.98 * len(prog))]),
    "progresion negativa": int(np.argsort(prog)[int(0.02 * len(prog))]),
}

cols = ["dist_media", "ctl_final", "ritmo_mediano", "semanas_act", "progresion"]
OUT["cualitativo"] = {}
for nombre, q in casos.items():
    _, R = ix_hnsw.search(X[q:q+1], K + 1)
    vec = [v for v in R[0] if v != q]      # los K vecinos, sin el query
    muestra = vec[:5]                      # los 5 primeros, para la tabla
    base = perfil.iloc[q]
    print(f"\n    {nombre.upper()} — atleta {ids[q]} ({ciudad[q]}, {gen[q]}, {edad[q]})")
    print(f"    {'':>6} {'dist':>7} {'CTL':>8} {'ritmo':>7} {'sem':>5} {'prog':>7}  destino")
    print(f"    {'query':>6} {base.dist_media:>7.1f} {base.ctl_final:>8.1f} "
          f"{base.ritmo_mediano:>7.2f} {base.semanas_act:>5.0f} "
          f"{base.progresion:>7.2f}  {ciudad[q]}")
    filas = []
    for rank, v in enumerate(muestra, 1):
        r = perfil.iloc[v]
        print(f"    {'#'+str(rank):>6} {r.dist_media:>7.1f} {r.ctl_final:>8.1f} "
              f"{r.ritmo_mediano:>7.2f} {r.semanas_act:>5.0f} "
              f"{r.progresion:>7.2f}  {ciudad[v]}")
        filas.append({c: float(r[c]) for c in cols} | {"ciudad": str(ciudad[v])})

    # Desviacion de los vecinos respecto del query, en unidades z.
    # Se calcula sobre los K vecinos, no sobre los 5 que muestra la
    # tabla, para que sea comparable con la referencia global de abajo.
    dev = float(np.abs(X[vec] - X[q]).mean())
    print(f"    desviacion media |z| de los {len(vec)} vecinos: {dev:.3f}")
    OUT["cualitativo"][nombre] = {
        "query": {c: float(base[c]) for c in cols} | {"ciudad": str(ciudad[q]),
                   "genero": str(gen[q]), "edad": str(edad[q])},
        "vecinos": filas, "dev_z": dev, "dev_z_k": int(len(vec)),
    }

# desviacion media global, como referencia
todos_dev = []
for q, v in zip(qi[:300], I_hn[:300]):
    todos_dev.append(np.abs(X[v] - X[q]).mean())
OUT["cualitativo"]["dev_z_global"] = float(np.mean(todos_dev))
print(f"\n    Desviacion media |z| global ({K} vecinos, 300 consultas): "
      f"{np.mean(todos_dev):.3f}")


# ================================================================
# SALIDA
# ================================================================
perfil.to_csv("perfiles.csv")
np.savez("indice_params.npz", mu=mu, sigma=sigma,
         features=np.array(perfil.columns, dtype=object))
faiss.write_index(ix_hnsw, "indice_hnsw.faiss")
with open("resultados.json", "w") as f:
    json.dump(OUT, f, indent=2)

print(f"\n\nGuardado: perfiles.csv, indice_hnsw.faiss, "
      f"indice_params.npz, resultados.json")
print("=" * 68)
