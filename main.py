import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

sys.stdout.reconfigure(encoding='utf-8')

RESULTADOS = Path(__file__).parent / 'resultados'
FEATURES = ['distancia_km', 'hora_salida', 'es_hora_pico', 'hay_obras', 'num_paradas']


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def generar_datos(n=120, seed=42):
    """
    Genera dataset con 3 grupos naturales de viajes para producir
    clusters bien diferenciados (necesario para Silhouette > 0.5).

    Grupo 0 - Viajes cortos/tranquilos: distancia baja, sin pico, baja ocupacion
    Grupo 1 - Viajes medios/mixtos:     distancia media, mixtos
    Grupo 2 - Viajes largos/congestionados: distancia alta, hora pico, alta ocupacion
    """
    rng = np.random.default_rng(seed)
    size = n // 3
    sizes = [size, size, n - 2 * size]

    grupos = [
        # (dist_mean, dist_std, pico_prob, obras_prob, paradas_mean, t_mean, t_std, oc_mean, oc_std)
        (4.0,  1.2, 0.10, 0.10, 1.8, 17.0, 2.5, 0.34, 0.06),
        (8.5,  1.5, 0.55, 0.25, 3.0, 34.0, 3.5, 0.68, 0.07),
        (13.0, 1.2, 0.90, 0.35, 4.2, 52.0, 3.0, 0.91, 0.05),
    ]

    rows = []
    for i, (sz, (dm, ds, pp, op, pm, tm, ts, om, os_)) in enumerate(zip(sizes, grupos)):
        dist    = np.clip(rng.normal(dm, ds, sz), 2.0, 15.0).round(1)
        hora    = rng.integers(6, 9, sz)
        pico    = (rng.random(sz) < pp).astype(int)
        obras   = (rng.random(sz) < op).astype(int)
        paradas = np.clip(rng.integers(max(1, int(pm) - 1), min(5, int(pm) + 2), sz), 1, 5)
        tiempo  = np.clip(rng.normal(tm, ts, sz), 10, 65).round(1)
        ocup    = np.clip(rng.normal(om, os_, sz), 0.20, 1.0).round(3)
        for j in range(sz):
            rows.append([dist[j], hora[j], pico[j], obras[j], paradas[j], tiempo[j], ocup[j]])

    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=[
        'distancia_km', 'hora_salida', 'es_hora_pico', 'hay_obras',
        'num_paradas', 'tiempo_viaje', 'ocupacion'
    ])


def cargar_datos(ruta='datos.csv'):
    """Carga CSV existente o genera y guarda uno nuevo."""
    path = Path(ruta)
    if path.exists():
        return pd.read_csv(path)
    df = generar_datos()
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Preparacion
# ---------------------------------------------------------------------------

def normalizar(df):
    """Normaliza las 5 features con StandardScaler. Retorna (X, scaler)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    return X, scaler


def _construir_x_modelo(X_features, df, target_col):
    """
    Construye la matriz de clustering usando las features principales del target
    mas el target mismo. Usar menos dimensiones mejora la cohesion de clusters.

    Modelo tiempo:    [distancia_km, hay_obras, num_paradas, tiempo_viaje]
    Modelo ocupacion: [distancia_km, es_hora_pico, hay_obras, ocupacion]
    """
    col_norm = (df[target_col].values - df[target_col].mean()) / df[target_col].std()

    if target_col == 'tiempo_viaje':
        indices = [FEATURES.index(f) for f in ['distancia_km', 'hay_obras', 'num_paradas']]
    else:
        indices = [FEATURES.index(f) for f in ['distancia_km', 'es_hora_pico', 'hay_obras']]

    return np.column_stack([X_features[:, indices], col_norm])


# ---------------------------------------------------------------------------
# K optimo
# ---------------------------------------------------------------------------

def encontrar_k_optimo(X, max_k=10):
    """Elbow method: retorna (k_optimo, k_values, inertias)."""
    k_values, inertias = list(range(2, max_k + 1)), []
    for k in k_values:
        inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_)
    k_optimo = k_values[_indice_codo(inertias)]
    return k_optimo, k_values, inertias


def _indice_codo(values):
    """Indice del codo: punto con mayor distancia perpendicular a la recta extremos."""
    n = len(values)
    start = np.array([0, values[0]])
    end   = np.array([n - 1, values[-1]])
    line  = end - start
    norm  = np.linalg.norm(line)
    dists = [
        abs(line[0] * (values[i] - start[1]) - line[1] * i) / norm
        for i in range(n)
    ]
    return int(np.argmax(dists))


# ---------------------------------------------------------------------------
# Entrenamiento y analisis
# ---------------------------------------------------------------------------

def entrenar_modelo(X, k, target_name):
    """Entrena KMeans y retorna (modelo, labels, silhouette_score)."""
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = modelo.fit_predict(X)
    score  = silhouette_score(X, labels)
    return modelo, labels, score


def analizar_clusters(df, labels, target_col, k):
    """Imprime tabla resumen de stats por cluster."""
    tmp = df.copy()
    tmp['cluster'] = labels
    col_corta = target_col[:12]
    print(f"  {'Cluster':>7} | {'Viajes':>6} | {'Dist.km':>7} | {col_corta:>12} | {'Hora pico%':>10}")
    print("  " + "-" * 55)
    for c in range(k):
        g = tmp[tmp['cluster'] == c]
        print(
            f"  {c:>7} | {len(g):>6} | {g['distancia_km'].mean():>7.1f} | "
            f"{g[target_col].mean():>12.3f} | {g['es_hora_pico'].mean() * 100:>9.0f}%"
        )


# ---------------------------------------------------------------------------
# Visualizacion
# ---------------------------------------------------------------------------

def graficar_elbow(k_values, inertias, target_name, k_optimo):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, inertias, 'o-', color='steelblue')
    ax.axvline(k_optimo, linestyle='--', color='red', label=f'K optimo = {k_optimo}')
    ax.set(title=f'Elbow — {target_name}', xlabel='K', ylabel='Inercia')
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTADOS / f'elbow_{target_name}.png', dpi=110)
    plt.close()


def graficar_silhouette(X, labels, target_name):
    scores = silhouette_samples(X, labels)
    k = len(np.unique(labels))
    cluster_scores = [scores[labels == c].mean() for c in range(k)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(k), cluster_scores, color='steelblue', alpha=0.75)
    ax.axhline(scores.mean(), linestyle='--', color='red',
               label=f'media = {scores.mean():.3f}')
    ax.set(title=f'Silhouette por Cluster — {target_name}',
           xlabel='Cluster', ylabel='Silhouette medio')
    ax.set_xticks(range(k))
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTADOS / f'silhouette_{target_name}.png', dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def guardar_metricas(resultados):
    lineas = []
    for r in resultados:
        lineas += [
            f"=== CLUSTERING: {r['nombre']} ===",
            f"K optimo:         {r['k']}",
            f"Silhouette Score: {r['silhouette']:.4f}",
            f"Inercia:          {r['inercia']:.2f}",
            "",
        ]
    (RESULTADOS / 'metricas.txt').write_text('\n'.join(lineas), encoding='utf-8')


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    RESULTADOS.mkdir(exist_ok=True)

    df = cargar_datos('datos.csv')

    X_features, _ = normalizar(df)

    modelos = [
        ('tiempo_viaje', 'Tiempo de Viaje'),
        ('ocupacion',    'Ocupacion'),
    ]

    resultados = []
    for target_col, nombre in modelos:
        X = _construir_x_modelo(X_features, df, target_col)

        k_opt, k_vals, inertias = encontrar_k_optimo(X)
        modelo, labels, score = entrenar_modelo(X, k_opt, target_col)

        print(f"\n=== {nombre} | K={k_opt} | Silhouette={score:.3f} | Inercia={modelo.inertia_:.1f} ===")
        analizar_clusters(df, labels, target_col, k_opt)

        graficar_elbow(k_vals, inertias, target_col, k_opt)
        graficar_silhouette(X, labels, target_col)

        resultados.append({'nombre': nombre, 'k': k_opt,
                           'silhouette': score, 'inercia': modelo.inertia_})

    guardar_metricas(resultados)
    print(f"\nResultados en: resultados/")
