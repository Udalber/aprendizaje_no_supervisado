# Tarea 4 — Clustering No Supervisado (K-Means)

**Estudiante:** Ángel Udalber Rodríguez Moya (100216147)  
**Curso:** Inteligencia Artificial — IBERO 2026  
**Dominio:** Sistema de Transporte SITP/TransMilenio, Bogotá

## ¿Qué hace?

Agrupa 120 viajes en clusters usando **K-Means** sin supervisión, descubriendo patrones naturales de viajes.

Entrena 2 modelos de clustering independientes:
1. **Modelo A:** agrupa por patrón de `tiempo_viaje`
2. **Modelo B:** agrupa por patrón de `ocupacion`

## Requisitos

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Cómo ejecutar

```bash
python main.py        # Ejecuta pipeline completo (~8 segundos)
python test_clustering.py    # Valida funcionalidad (3 tests)
```

## Archivos generados

**En `resultados/`:**
- `metricas.txt` — K óptimo, Silhouette Score, Inercia
- `elbow_tiempo_viaje.png` — Gráfica Elbow para Modelo A
- `elbow_ocupacion.png` — Gráfica Elbow para Modelo B
- `silhouette_tiempo_viaje.png` — Cohesión por cluster (Modelo A)
- `silhouette_ocupacion.png` — Cohesión por cluster (Modelo B)

**En raíz:**
- `datos.csv` — Dataset de 120 viajes (3 grupos naturales)

## Resultados esperados

```
=== Tiempo de Viaje | K=5 | Silhouette=0.551 ===
  Cluster | Viajes | Dist.km | tiempo_viaje | Hora pico%
  -------------------------------------------------------
        0 |     34 |     8.5 |       34.350 |        44%
        1 |     30 |    13.1 |       51.523 |        90%
        2 |      9 |    12.7 |       52.533 |       100%
        3 |     35 |     4.0 |       17.269 |         9%
        4 |     12 |     7.1 |       26.050 |        67%

=== Ocupacion | K=5 | Silhouette=0.613 ===
  Cluster | Viajes | Dist.km |    ocupacion | Hora pico%
  -------------------------------------------------------
        0 |     22 |     9.4 |        0.714 |         0%
        1 |     35 |     4.1 |        0.336 |         0%
        2 |     30 |    12.9 |        0.893 |       100%
        3 |     18 |    10.2 |        0.744 |        94%
        4 |     15 |     6.5 |        0.601 |       100%
```

**Métricas de calidad:**
- Silhouette 0.55 y 0.61 → **EXCELENTE** (> 0.5)
- Clusters bien definidos y separados

## Estructura del código

### `main.py`
Pipeline completo:
1. Genera 120 viajes en 3 grupos naturales
2. Normaliza con StandardScaler
3. Para cada target:
   - Encuentra K óptimo (Elbow Method)
   - Entrena KMeans
   - Calcula Silhouette Score
   - Visualiza resultados
4. Guarda métricas y gráficas

### `test_clustering.py`
3 tests básicos:
- Validar generación de datos (rangos, nulls)
- Validar normalización (media=0, std=1)
- Validar clustering (labels, silhouette score)

## Interpretación

### Elbow Method
Gráficas K vs Inercia. El codo (punto donde la pendiente cambia) indica el K óptimo.

### Silhouette Score
Métrica 0-1 que mide cohesión de clusters:
- **0.5+** = clusters bien definidos
- **0.3-0.5** = clusters presentes pero solapados
- **<0.3** = clusters débiles

### Tabla de análisis
Estadísticas por cluster (distancia, tiempo, ocupación, % hora pico) para interpretación en dominio.

## Diferencia: Regresión vs Clustering

| Aspecto | Regresión (Tarea 3) | Clustering (Tarea 4) |
|---------|-----------|-----------|
| Objetivo | Predecir valores | Agrupar similares |
| Supervisor | Necesita labels | Sin labels |
| Salida | Número (35 min) | Etiqueta de grupo |
| Métrica | R² | Silhouette |
| Algoritmo | LinearRegression | KMeans |

## Archivos incluidos

```
tarea4/
├── main.py                    (320 líneas, todo el pipeline)
├── test_clustering.py         (51 líneas, 3 tests)
├── README.md                  (este archivo)
├── .gitignore
├── datos.csv                  (generado: 120 registros)
└── resultados/                (generado automáticamente)
    ├── metricas.txt
    ├── elbow_tiempo_viaje.png
    ├── elbow_ocupacion.png
    ├── silhouette_tiempo_viaje.png
    └── silhouette_ocupacion.png
```

## Notas técnicas

- **Reproducibilidad:** `random_state=42` en todos los algoritmos
- **Sin data leakage:** normalización solo en train
- **Reutilización:** mismo pipeline para ambos targets
- **Modular:** funciones pequeñas, fáciles de entender
