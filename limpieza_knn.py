"""
Práctica 3: Implementación de Limpieza de Datos Avanzada en Python.

Flujo del script:
    1. Genera un dataset con ruido sintético (valores faltantes y atípicos).
    2. Análisis inicial de calidad: tabla de nulos y detección de outliers por IQR.
    3. Tratamiento: los outliers se reemplazan por NaN (IQR para ingreso y
       regla de dominio para edad > 100).
    4. Imputación multivariada con KNNImputer (k = 2).
    5. Experimento: varía k (1, 2, 3, 5) y compara contra media y mediana;
       además prueba KNN con variables escaladas.
    6. Gráficas antes/después (boxplots, histogramas, mapa de nulos y comparativa).

Todas las salidas (gráficas PNG y tablas CSV) se guardan en la carpeta "salidas/".

Uso:
    python limpieza_knn.py            # genera resultados y gráficas
    python limpieza_knn.py --mostrar  # además abre las gráficas en ventanas
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

CARPETA_SALIDAS = Path(__file__).parent / "salidas"
EDAD_MAXIMA_VALIDA = 100   # regla de dominio: una edad mayor a 100 se considera error de captura
K_BASE = 2                 # número de vecinos solicitado en la práctica
VALORES_K = [1, 2, 3, 5]   # valores de k para el experimento

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")


# ----------------------------------------------------------------------
# 1. Dataset con ruido sintético (código base)
# ----------------------------------------------------------------------
def generar_dataset():
    """Crea el dataset ruidoso del código base."""
    data = {
        "edad": [25, 30, np.nan, 45, 22, 120, 35, 28, np.nan, 40],  # 120 es atípico/erróneo
        "ingreso_mensual": [15000, 22000, 18000, np.nan, 12000, 950000,
                            25000, 21000, 19000, np.nan],             # 950k es outlier
        "scoring_credito": [650, 700, 680, 710, 590, 800, 670, 640, 620, 730],
    }
    return pd.DataFrame(data)


# ----------------------------------------------------------------------
# 2. Análisis de calidad
# ----------------------------------------------------------------------
def tabla_nulos(df):
    """Cantidad y porcentaje de valores faltantes por columna."""
    return pd.DataFrame({
        "nulos": df.isna().sum(),
        "%_nulos": (df.isna().mean() * 100).round(1),
        "filas_con_nulo": [list(df.index[df[c].isna()]) for c in df.columns],
    })


def limites_iqr(serie, factor=1.5):
    """Calcula Q1, Q3, IQR y los límites inferior/superior (ignora NaN)."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "limite_inferior": q1 - factor * iqr,
        "limite_superior": q3 + factor * iqr,
    }


def tabla_outliers_iqr(df):
    """Aplica IQR a cada columna y lista qué filas quedan fuera de los límites."""
    filas = {}
    for col in df.columns:
        lim = limites_iqr(df[col])
        fuera = df[(df[col] < lim["limite_inferior"]) | (df[col] > lim["limite_superior"])]
        # Verificación del IQR con scipy
        lim["IQR_scipy"] = stats.iqr(df[col], nan_policy="omit")
        lim["outliers_filas"] = list(fuera.index)
        lim["outliers_valores"] = list(fuera[col])
        filas[col] = lim
    return pd.DataFrame(filas).T


def zscore_columna(serie):
    """Z-score de cada valor (método complementario al IQR)."""
    z = stats.zscore(serie, nan_policy="omit")
    return pd.Series(z, index=serie.index).round(2)


# ----------------------------------------------------------------------
# 3. Tratamiento de outliers
# ----------------------------------------------------------------------
def tratar_outliers(df):
    """Reemplaza outliers por NaN para que después los estime el imputador."""
    df = df.copy()

    lim = limites_iqr(df["ingreso_mensual"])
    print(f"Límite Superior para Outliers en Ingreso: ${lim['limite_superior']:,.2f}")
    mascara_ingreso = df["ingreso_mensual"] > lim["limite_superior"]
    mascara_edad = df["edad"] > EDAD_MAXIMA_VALIDA

    print(f"Filas con ingreso atípico -> NaN: {list(df.index[mascara_ingreso])}")
    print(f"Filas con edad > {EDAD_MAXIMA_VALIDA} -> NaN: {list(df.index[mascara_edad])}")

    df.loc[mascara_ingreso, "ingreso_mensual"] = np.nan
    df.loc[mascara_edad, "edad"] = np.nan
    return df


# ----------------------------------------------------------------------
# 4. Imputación
# ----------------------------------------------------------------------
def imputar_knn(df, k, escalar=False):
    """KNNImputer con k vecinos. Si escalar=True, estandariza antes de medir distancias."""
    if not escalar:
        imputer = KNNImputer(n_neighbors=k)
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

    scaler = StandardScaler()                     # StandardScaler ignora los NaN al ajustar
    escalado = scaler.fit_transform(df)
    imputado = KNNImputer(n_neighbors=k).fit_transform(escalado)
    return pd.DataFrame(scaler.inverse_transform(imputado), columns=df.columns, index=df.index)


def imputar_simple(df, estrategia):
    """Imputación tradicional por media o mediana de cada columna."""
    imputer = SimpleImputer(strategy=estrategia)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)


def vecinos_usados(df, fila, columna, k):
    """Muestra qué filas usa KNNImputer como vecinos para una celda (distancia nan_euclidean)."""
    from sklearn.metrics.pairwise import nan_euclidean_distances

    candidatos = df.index[df[columna].notna() & (df.index != fila)]
    dist = nan_euclidean_distances(df.loc[[fila]], df.loc[candidatos])[0]
    orden = pd.Series(dist, index=candidatos).sort_values()
    vecinos = orden.head(k)
    return pd.DataFrame({
        "distancia": vecinos.round(2),
        columna: df.loc[vecinos.index, columna],
    })


def comparar_metodos(df_tratado):
    """Tabla con el valor imputado en cada celda faltante según el método."""
    celdas = [(i, c) for c in df_tratado.columns for i in df_tratado.index[df_tratado[c].isna()]]

    metodos = {"Media": imputar_simple(df_tratado, "mean"),
               "Mediana": imputar_simple(df_tratado, "median")}
    for k in VALORES_K:
        metodos[f"KNN k={k}"] = imputar_knn(df_tratado, k)
    metodos[f"KNN k={K_BASE} escalado"] = imputar_knn(df_tratado, K_BASE, escalar=True)

    tabla = pd.DataFrame(
        {nombre: [res.loc[i, c] for i, c in celdas] for nombre, res in metodos.items()},
        index=[f"fila {i} · {c}" for i, c in celdas],
    )
    return tabla, metodos


# ----------------------------------------------------------------------
# 5. Gráficas
# ----------------------------------------------------------------------
def guardar(fig, nombre):
    ruta = CARPETA_SALIDAS / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"Gráfica guardada: {ruta}")


def grafica_nulos(df):
    """Mapa de calidad del dataset original: faltantes (NaN) y atípicos (IQR)."""
    estado = np.zeros(df.shape)                 # 0 = válido, 1 = NaN, 2 = atípico
    estado[df.isna().values] = 1
    for j, col in enumerate(df.columns):
        lim = limites_iqr(df[col])
        fuera = (df[col] < lim["limite_inferior"]) | (df[col] > lim["limite_superior"])
        estado[fuera.values, j] = 2

    colores = matplotlib.colors.ListedColormap(["#FFFFFF", "#404040", "#D62728"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.imshow(estado, aspect="auto", cmap=colores, vmin=0, vmax=2)
    ax.set_xticks(range(df.shape[1]), df.columns)
    ax.set_yticks(range(df.shape[0]), [f"fila {i}" for i in df.index])
    for (i, j), e in np.ndenumerate(estado):
        valor = df.iat[i, j]
        texto = "NaN" if e == 1 else f"{valor:,.0f}"
        ax.text(j, i, texto, ha="center", va="center",
                color="white" if e else "black", fontweight="bold" if e else "normal")
    ax.set_xticks(np.arange(-0.5, df.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0]), minor=True)
    ax.grid(which="minor", color="#BBBBBB", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    leyenda = [matplotlib.patches.Patch(color="#404040", label="Faltante (NaN)"),
               matplotlib.patches.Patch(color="#D62728", label="Atípico (IQR)")]
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title("Mapa de calidad del dataset original")
    guardar(fig, "01_mapa_calidad.png")


def grafica_boxplots(df_original, df_final):
    """Boxplots antes y después del tratamiento de outliers."""
    fig, ejes = plt.subplots(2, 3, figsize=(13, 7))
    for j, col in enumerate(df_original.columns):
        for i, (titulo, datos) in enumerate([("Antes", df_original), ("Después", df_final)]):
            ax = ejes[i, j]
            ax.boxplot(datos[col].dropna(), widths=0.5,
                       flierprops=dict(marker="o", markerfacecolor="red", markersize=8))
            ax.set_title(f"{titulo} · {col}")
            ax.set_xticks([])
            ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    fig.suptitle("Boxplots antes y después del tratamiento de outliers (IQR) e imputación KNN (k=2)")
    fig.tight_layout()
    guardar(fig, "02_boxplots_antes_despues.png")


def grafica_histogramas(df_original, df_final):
    """Histogramas de ingreso y edad antes y después."""
    fig, ejes = plt.subplots(2, 2, figsize=(12, 7))
    for j, col in enumerate(["ingreso_mensual", "edad"]):
        ejes[0, j].hist(df_original[col].dropna(), bins=10, color="grey", edgecolor="black")
        ejes[0, j].set_title(f"Antes · {col}")
        ejes[1, j].hist(df_final[col], bins=10, color="steelblue", edgecolor="black")
        ejes[1, j].set_title(f"Después · {col}")
        for ax in ejes[:, j]:
            ax.set_ylabel("frecuencia")
            ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    fig.suptitle("Distribución antes y después de la limpieza")
    fig.tight_layout()
    guardar(fig, "03_histogramas_antes_despues.png")


def grafica_comparativa(tabla):
    """Barras con el valor imputado por cada método, una gráfica por variable."""
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(ejes, ["edad", "ingreso_mensual"]):
        sub = tabla[tabla.index.str.endswith(col)]
        sub.T.plot(kind="bar", ax=ax, width=0.8, edgecolor="black")
        ax.set_title(f"Valores imputados de '{col}' según el método")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.legend(title="celda imputada", fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.36), ncol=3)
    fig.tight_layout()
    guardar(fig, "04_comparativa_metodos.png")


# ----------------------------------------------------------------------
# 6. Programa principal
# ----------------------------------------------------------------------
def main(mostrar=False):
    if not mostrar:
        matplotlib.use("Agg")  # solo guarda archivos, no abre ventanas
    CARPETA_SALIDAS.mkdir(exist_ok=True)

    df_dirty = generar_dataset()
    df_original = df_dirty.copy()

    print("=== DATASET ORIGINAL (RUIDOSO) ===")
    print(df_original)

    print("\n=== 1. ANÁLISIS INICIAL DE CALIDAD ===")
    print("\nValores faltantes por columna:")
    print(tabla_nulos(df_original))

    print("\nDetección de outliers por IQR (factor 1.5):")
    t_iqr = tabla_outliers_iqr(df_original)
    print(t_iqr)

    print("\nZ-score de ingreso_mensual (complementario):")
    print(zscore_columna(df_original["ingreso_mensual"]).to_string())

    print("\nEstadísticos descriptivos originales:")
    print(df_original.describe().T)

    print("\n=== 2. TRATAMIENTO DE OUTLIERS ===")
    df_tratado = tratar_outliers(df_dirty)
    print("\nDataset con outliers reemplazados por NaN:")
    print(df_tratado)

    print(f"\n=== 3. IMPUTACIÓN CON KNNImputer (k={K_BASE}) ===")
    print("Vecinos usados para imputar 'ingreso_mensual' de la fila 3:")
    print(vecinos_usados(df_tratado, 3, "ingreso_mensual", K_BASE))
    df_clean = imputar_knn(df_tratado, K_BASE)
    print("\n=== DATASET LIMPIO E IMPUTADO ===")
    print(df_clean)

    print("\n=== 4. COMPARATIVA DE MÉTODOS DE IMPUTACIÓN ===")
    tabla, metodos = comparar_metodos(df_tratado)
    print(tabla)

    resumen = pd.DataFrame({
        "Original (con ruido)": df_original.mean(),
        "Media": metodos["Media"].mean(),
        "Mediana": metodos["Mediana"].mean(),
        f"KNN k={K_BASE}": df_clean.mean(),
    })
    print("\nPromedio de cada variable según el tratamiento:")
    print(resumen)

    # Guardar tablas
    df_original.to_csv(CARPETA_SALIDAS / "dataset_original.csv", index=False)
    df_clean.to_csv(CARPETA_SALIDAS / "dataset_limpio_knn_k2.csv", index=False)
    tabla.to_csv(CARPETA_SALIDAS / "comparativa_metodos.csv")

    print("\n=== 5. GRÁFICAS ===")
    grafica_nulos(df_original)
    grafica_boxplots(df_original, df_clean)
    grafica_histogramas(df_original, df_clean)
    grafica_comparativa(tabla)

    if mostrar:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limpieza de datos con IQR y KNNImputer")
    parser.add_argument("--mostrar", action="store_true", help="abre las gráficas en ventanas")
    main(parser.parse_args().mostrar)
