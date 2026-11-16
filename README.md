# Recuperación por similitud de perfiles de entrenamiento

Prueba de concepto del Trabajo Final de Grado *Aplicación de bases de
datos vectoriales a la recuperación por similitud de perfiles de
entrenamiento: caso de estudio sobre corredores amateurs de maratón*
(Federico García Monzón, Universidad de Palermo).

Indexa 14.612 corredores de maratón en FAISS —un índice exacto
`IndexFlatL2` como referencia y uno aproximado `IndexHNSWFlat`— y
resuelve consultas de vecinos más próximos sobre perfiles de
entrenamiento de 13 dimensiones, incluidos corredores externos al
corpus.

Los perfiles se derivan del conjunto público *A public dataset on
long-distance running training in 2019 and 2020* (Afonseca, Watanabe y
Duarte, Figshare 2021,
[10.6084/m9.figshare.16620238](https://doi.org/10.6084/m9.figshare.16620238)).

`resultados.json` contiene todas las cifras que reporta el paper,
calculadas por `pipeline.py`. El apartado [Qué contiene
`resultados.json`](#qué-contiene-resultadosjson) explica cuáles se
reproducen de forma exacta y cuáles dependen del equipo.

## Instalación (una vez)

```
pip install streamlit faiss-cpu pandas numpy
```

Python 3.9 o superior. No hace falta GPU. `faiss-cpu`, no `faiss` a
secas.

**Importante:** todos los comandos se corren *desde esta carpeta*. Los
programas buscan el índice y los parámetros en el directorio actual.

## Interfaz gráfica — la forma recomendada de verla

```
streamlit run app.py
```

Abre en el navegador. En la barra lateral se arma el corredor a
consultar, con los deslizadores de kilómetros, ritmo, semanas y
tendencia del volumen, o se sube un CSV con un historial real. La página
muestra el perfil calculado, los vecinos recuperados y la comparación de
rangos entre la consulta y su vecindad.

El índice ya viene construido en este paquete, así que responde en el
momento: no reconstruye nada.

### Consola, como alternativa

La interfaz y la consola comparten exactamente el mismo cálculo. Si
Streamlit no arranca, la consola hace lo mismo:

```
python consultar.py
python consultar.py --km 80 --ritmo 4.4 --semanas 45
python consultar.py --km 18 --ritmo 6.8 --semanas 22 --progresion baja
```

| Parámetro | Qué es |
|---|---|
| `--km` | promedio de kilómetros por semana |
| `--ritmo` | ritmo típico en min/km |
| `--semanas` | cuántas semanas entrenó |
| `--progresion` | `sube`, `estable` o `baja` |

### Con un historial real

```
python consultar.py mi_historial.csv
```

`mi_historial.csv` viene incluido como ejemplo: son 30 semanas de una
preparación de maratón, con bloques de carga, semanas de descarga, una
semana perdida y tres de afinamiento. Sirve para mostrar un caso que no
es sintético.

El CSV necesita dos columnas, una fila por semana en orden cronológico:

```
distance,duration
32.4,178.2
35.1,192.0
...
```

## Reconstruir todo desde cero — unos 4 minutos

Para verificar el pipeline completo hace falta `run_ww_2019_w.csv`
(142 MB, se baja de Figshare, DOI 10.6084/m9.figshare.16620238).

```
python pipeline.py run_ww_2019_w.csv
```

Corre las diez etapas en una sola ejecución: carga y filtrado,
caracterización del corpus, métricas de carga, construcción de los
vectores, los dos índices, recall, proxy semántico, barrido de
efSearch, escalado e inspección cualitativa. Regenera
`indice_hnsw.faiss`, `indice_params.npz`, `perfiles.csv` y
`resultados.json`.

Para las figuras del paper:

```
pip install matplotlib
python figuras.py
```

## Qué contiene `resultados.json`

Todas las cifras que reporta el paper, calculadas por el programa y no
transcritas a mano. Por bloque:

| Bloque | Qué guarda |
|---|---|
| `corpus` | la cadena de filtrado, de 36.412 atletas a 14.612 |
| `descriptivo` | distancia semanal mediana y p95, registros retenidos, composición demográfica, países y distribución por maratón objetivo |
| `carga` | medias y medianas de TSS y CTL |
| `vector` | dimensión, cantidad de vectores y tamaño de la matriz |
| `construccion` | tiempos y memoria de los dos índices |
| `busqueda` | latencias, aceleración y recall@10 |
| `proxy` | coincidencia observada y esperada por azar, con su lift |
| `barrido` | recall y latencia para cada valor de efSearch |
| `escalado` | las cuatro réplicas del corpus, con su aceleración y recall |
| `cualitativo` | los cuatro casos extremos, sus vecinos y la desviación media en unidades z |

Dos clases de cifra, que conviene no confundir al verificar:

- **Deterministas.** Los conteos, los recall y las tasas de coincidencia
  salen siempre iguales, porque todos los muestreos usan una semilla
  fija (`SEED = 42`). Se reproducen exactamente en cualquier máquina.
- **Latencias.** Dependen del equipo. Entre máquinas distintas lo
  comparable es la relación entre tiempos, no el valor absoluto. Los
  números del paper se midieron sobre un núcleo, sin GPU, con
  `faiss-cpu` 1.15.

La desviación media en unidades z se calcula sobre los 10 vecinos
recuperados, los mismos que usa la referencia global; la tabla del paper
muestra solo los 3 primeros por espacio.

## Qué hay en cada archivo

| Archivo | Qué es |
|---|---|
| `app.py` | Interfaz Streamlit |
| `consultar.py` | La misma consulta, por consola |
| `pipeline.py` | Pipeline completo y los experimentos del paper |
| `figuras.py` | Genera las dos figuras |
| `indice_hnsw.faiss` | Índice ya construido (14.612 vectores) |
| `indice_params.npz` | Medias y desvíos del corpus, más los nombres de las features |
| `perfiles.csv` | Los 14.612 perfiles con sus 13 dimensiones |
| `resultados.json` | Todas las cifras que cita el paper |
| `mi_historial.csv` | Historial de ejemplo, 30 semanas |

## Un detalle que importa

`indice_params.npz` guarda la media y el desvío **del corpus**. Un
corredor externo se normaliza con esos valores, no con los suyos: si se
recalculara el z-score sobre su propio historial, el vector quedaría en
otra escala y las distancias no significarían nada.

Por eso el índice y esos parámetros se persisten juntos.

## Si algo falla

| Error | Qué hacer |
|---|---|
| `Faltan archivos en esta carpeta: ...` | El comando se corrió desde otro directorio. Hacer `cd` a esta carpeta y repetirlo |
| `Falta faiss` | `pip install faiss-cpu` (no `faiss` a secas) |
| `streamlit: command not found` | `pip install streamlit`, o usar `python -m streamlit run app.py` |
| `Al CSV le faltan columnas` | El CSV necesita `distance` y `duration` |
