# Práctica 3: Limpieza de Datos Avanzada en Python

Script que identifica datos faltantes y atípicos en un dataset con ruido
sintético, detecta outliers con el **Rango Intercuartílico (IQR)**, los
reemplaza por `NaN` y los imputa con **KNNImputer** de scikit-learn (k = 2).
También compara distintos valores de k contra la imputación por media y
mediana, y genera gráficas antes/después.

## Estructura

| Archivo / carpeta | Descripción |
|---|---|
| `limpieza_knn.py` | Script principal: análisis de calidad, IQR, KNNImputer, comparativa y gráficas |
| `salidas/` | Gráficas PNG y tablas CSV generadas por el script |
| `requirements.txt` | Dependencias del proyecto |
| `.gitignore` | Excluye entornos virtuales, cachés y credenciales |

## Instalación y ejecución

```bash
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
python limpieza_knn.py             # genera resultados en salidas/
python limpieza_knn.py --mostrar   # además abre las gráficas
```

## Proceso

1. **Análisis inicial de calidad**: tabla de nulos por columna y límites IQR
   (Q1, Q3, IQR, límite inferior y superior) para cada variable; z-score
   como método complementario.
2. **Tratamiento de outliers**: el ingreso mayor a `Q3 + 1.5·IQR` y la edad
   mayor a 100 años se reemplazan por `NaN`.
3. **Imputación KNN (k = 2)**: cada `NaN` se estima con el promedio de los
   2 registros más parecidos según la distancia `nan_euclidean`.
4. **Experimento**: k = 1, 2, 3 y 5, media, mediana y KNN con variables
   escaladas (`StandardScaler`).

## Resultados principales

| | Valor |
|---|---|
| Límite superior IQR de ingreso | $31,000.00 |
| Outliers detectados | fila 5: edad 120 e ingreso 950,000 |
| Faltantes originales | 4 (edad: filas 2 y 8; ingreso: filas 3 y 9) |
| Celdas imputadas tras el tratamiento | 6 |

## Gráficas generadas

- `01_mapa_calidad.png`: faltantes y atípicos del dataset original
- `02_boxplots_antes_despues.png`
- `03_histogramas_antes_despues.png`
- `04_comparativa_metodos.png`: valores imputados por cada método
