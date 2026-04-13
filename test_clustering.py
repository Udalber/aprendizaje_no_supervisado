"""
Tests para Tarea 4 — Clustering K-Means

Ejecucion: python test_clustering.py
           pytest test_clustering.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding='utf-8')

from main import generar_datos, normalizar, entrenar_modelo


def test_generacion_datos():
    df = generar_datos(n=50)
    assert len(df) == 50
    assert df.isnull().sum().sum() == 0
    assert list(df.columns) == [
        'distancia_km', 'hora_salida', 'es_hora_pico',
        'hay_obras', 'num_paradas', 'tiempo_viaje', 'ocupacion'
    ]
    assert df['distancia_km'].between(2.0, 15.0).all()
    assert df['tiempo_viaje'].between(10, 65).all()
    assert df['ocupacion'].between(0.20, 1.0).all()


def test_normalizacion():
    df = generar_datos(n=50)
    X, scaler = normalizar(df)
    assert X.shape == (50, 5)
    assert -4 < X.mean() < 4
    assert abs(X.std() - 1.0) < 0.1


def test_clustering():
    df = generar_datos(n=50)
    X, _ = normalizar(df)
    modelo, labels, score = entrenar_modelo(X, k=3, target_name='test')
    assert len(labels) == 50
    assert set(labels) == {0, 1, 2}
    assert 0 <= score <= 1


if __name__ == '__main__':
    test_generacion_datos()
    test_normalizacion()
    test_clustering()
    print("Todos los tests pasaron")
