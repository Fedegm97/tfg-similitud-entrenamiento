"""
Interfaz de la prueba de concepto
==================================
TFG · Federico Garcia Monzon

Envuelve la logica de consultar.py en una interfaz web, para poder
consultar el indice en vivo sin pasar por la terminal.

    pip install streamlit faiss-cpu pandas numpy
    streamlit run app.py

Necesita en la misma carpeta, igual que consultar.py:
    indice_hnsw.faiss, indice_params.npz, perfiles.csv

La logica de generacion del historial, el calculo de las trece
dimensiones y la estandarizacion son identicas a consultar.py: ante los
mismos parametros, las dos vias devuelven exactamente los mismos vecinos.
"""

import inspect
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Recuperación por similitud — PoC",
                   page_icon="•", layout="wide")

# st.dataframe cambio de parametro: use_container_width quedo deprecado
# a favor de width. Se elige el que acepte la version instalada, asi la
# interfaz anda tanto en una Streamlit vieja como en una nueva.
ANCHO = ({"width": "stretch"}
         if "width" in inspect.signature(st.dataframe).parameters
         else {"use_container_width": True})

# Tipografia algo mas grande: esto se proyecta en un aula.
st.markdown("""
<style>
  html, body, [class*="css"] { font-size: 17px; }
  [data-testid="stMetricValue"] { font-size: 30px; }
  [data-testid="stMetricLabel"] { font-size: 15px; }
  thead tr th { font-size: 15px !important; }
  tbody tr td { font-size: 16px !important; }
</style>
""", unsafe_allow_html=True)

REQUERIDOS = ["indice_hnsw.faiss", "indice_params.npz", "perfiles.csv"]


# ---------------------------------------------------------------
# Carga del indice (una sola vez)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando el índice…")
def cargar():
    import faiss
    ix = faiss.read_index("indice_hnsw.faiss")
    p = np.load("indice_params.npz", allow_pickle=True)
    perfiles = pd.read_csv("perfiles.csv", index_col=0)
    return ix, p["mu"], p["sigma"], list(p["features"]), perfiles


faltan = [f for f in REQUERIDOS if not os.path.exists(f)]
if faltan:
    st.error(f"Faltan archivos en esta carpeta: {', '.join(faltan)}")
    st.caption("Son los que genera pipeline.py y vienen en poc_tfg.zip. "
               "Copialos junto a app.py.")
    st.stop()

try:
    ix, mu, sigma, feats, perfiles = cargar()
except ImportError:
    st.error("Falta faiss. Instalalo con:  pip install faiss-cpu")
    st.stop()


# ---------------------------------------------------------------
# Misma logica que consultar.py — no tocar sin tocar alla tambien
# ---------------------------------------------------------------
def historial_sintetico(km, ritmo, semanas, tendencia):
    """Historial con variacion semana a semana, calibrado para que el
    promedio de distancia y la mediana de ritmo den exactamente lo pedido."""
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
    return pd.DataFrame({"distance": dist, "duration": np.round(dist * r, 1)})


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

    # Clave: se normaliza con los estadisticos del CORPUS, no con los del
    # propio atleta. De otro modo el vector quedaria en otra escala.
    return ((x - mu) / (sigma + 1e-9)).astype(np.float32).reshape(1, -1), v


# ---------------------------------------------------------------
# Controles
# ---------------------------------------------------------------
st.title("¿Qué corredores se parecen a este?")
st.caption(f"Índice HNSW · {ix.ntotal:,} corredores · {ix.d} dimensiones por perfil"
           .replace(",", "."))

with st.sidebar:
    st.header("El corredor a consultar")
    fuente = st.radio("Historial", ["Generado", "Desde un CSV"],
                      horizontal=True, label_visibility="collapsed")

    archivo = None
    if fuente == "Desde un CSV":
        archivo = st.file_uploader("CSV con columnas distance y duration",
                                   type="csv")
        km = ritmo = semanas = None
        progresion = "sube"
    else:
        km = st.slider("Kilómetros por semana", 5.0, 120.0, 40.0, 1.0)
        ritmo = st.slider("Ritmo (min/km)", 3.0, 9.0, 5.5, 0.1)
        semanas = st.slider("Semanas entrenadas", 20, 52, 30, 1)
        progresion = st.select_slider("Volumen",
                                      options=["baja", "estable", "sube"],
                                      value="sube")

    st.divider()
    k = st.slider("Vecinos a recuperar", 3, 10, 5, 1)


# ---------------------------------------------------------------
# Consulta
# ---------------------------------------------------------------
if fuente == "Desde un CSV":
    if archivo is None:
        st.info("Subí un CSV con una fila por semana, en orden cronológico, "
                "con las columnas **distance** (km) y **duration** (minutos).")
        st.stop()
    semanas_df = pd.read_csv(archivo)
    falta = {"distance", "duration"} - set(semanas_df.columns)
    if falta:
        st.error(f"Al CSV le faltan columnas: {', '.join(sorted(falta))}")
        st.stop()
    origen = f"{archivo.name} · {len(semanas_df)} semanas"
else:
    semanas_df = historial_sintetico(km, ritmo, semanas, progresion)
    origen = (f"{semanas} semanas · ~{km:.0f} km/sem · "
              f"ritmo {ritmo:.1f} · volumen {progresion}")

q, v = vectorizar(semanas_df)
D, I = ix.search(q, k)
vecinos = perfiles.iloc[I[0]]

st.caption(f"Consulta: {origen}")


# --- el perfil que se calculo -----------------------------------
st.subheader("Perfil calculado")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Km por semana", f"{v['dist_media']:.1f}")
c2.metric("Ritmo mediano", f"{v['ritmo_mediano']:.2f}")
c3.metric("CTL final", f"{v['ctl_final']:.0f}")
c4.metric("Progresión", f"{v['progresion']:+.2f}")
st.caption("Estas cuatro son 4 de las 13 dimensiones del vector. "
           "La búsqueda usa las trece.")


# --- los vecinos ------------------------------------------------
st.subheader(f"Los {k} corredores más parecidos del corpus")

tabla = pd.DataFrame({
    "Km/sem":     vecinos["dist_media"].round(1).values,
    "CTL final":  vecinos["ctl_final"].round(1).values,
    "Ritmo":      vecinos["ritmo_mediano"].round(2).values,
    "Semanas":    vecinos["semanas_act"].astype(int).values,
    "Progresión": vecinos["progresion"].round(2).values,
    "Distancia":  D[0].round(2),
}, index=[f"#{i}" for i in range(1, k + 1)])
st.dataframe(tabla, **ANCHO)

st.caption("**Distancia** es la distancia euclídea en el espacio "
           "estandarizado: más chica, más parecido.")


# --- el rango, que es lo que convence ---------------------------
st.subheader("Qué tan cerca quedaron")
r1, r2, r3 = st.columns(3)
r1.metric("Km/sem de los vecinos",
          f"{vecinos.dist_media.min():.1f} – {vecinos.dist_media.max():.1f}",
          f"consulta: {v['dist_media']:.1f}", delta_color="off")
r2.metric("Ritmo de los vecinos",
          f"{vecinos.ritmo_mediano.min():.2f} – {vecinos.ritmo_mediano.max():.2f}",
          f"consulta: {v['ritmo_mediano']:.2f}", delta_color="off")
r3.metric("Progresión de los vecinos",
          f"{vecinos.progresion.min():+.2f} – {vecinos.progresion.max():+.2f}",
          f"consulta: {v['progresion']:+.2f}", delta_color="off")

st.info("Ninguna de estas tres variables se usó como criterio de búsqueda. "
        "La cercanía se mide sobre el vector completo de trece dimensiones: "
        "que coincidan las tres a la vez es consecuencia de la representación, "
        "no de haber filtrado por ellas.")


# --- apoyos para la defensa -------------------------------------
if fuente == "Generado":
    with st.expander("El mismo resultado por línea de comandos"):
        st.code(f"python consultar.py --km {km:g} --ritmo {ritmo:g} "
                f"--semanas {semanas} --progresion {progresion}",
                language="bash")
        st.caption("Misma lógica, mismos vecinos. Si la interfaz falla, "
                   "este comando da lo mismo.")

with st.expander("Por qué el z-score se calcula con los datos del corpus"):
    st.markdown(
        "El corredor consultado se estandariza con la **media y el desvío "
        "del corpus**, no con los suyos.\n\n"
        "Si se recalculara el z-score sobre su propio historial, su vector "
        "quedaría en otra escala y las distancias no significarían nada: "
        "sería comparar alturas donde cada persona usa su propia regla.\n\n"
        "Por eso el índice se persiste junto con esos parámetros, en "
        "`indice_params.npz`.")

with st.expander("Qué prueba esto y qué no"):
    st.markdown(
        "**Prueba** que la representación discrimina: perfiles distintos "
        "recuperan vecindades distintas, y los vecinos reproducen el perfil "
        "de consulta en magnitudes que no fueron criterio de búsqueda.\n\n"
        "**No prueba** que esos vecinos sean los *correctos* en sentido "
        "deportivo. Para eso haría falta una noción de similitud etiquetada "
        "que el conjunto de datos no tiene. Es la limitación que el paper "
        "declara en las conclusiones.")
