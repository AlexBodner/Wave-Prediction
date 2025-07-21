import matplotlib.pyplot as plt
import utils.fit_mean_plane as fit_mean_plane
import pyvista as pv
import numpy as np
import open3d as o3d
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def generate_weighted_height_grid(points, grid_size, origin, max_coords, radius=0.1, power=2):
    """
    Genera una grilla de alturas a partir de una nube de puntos 3D, promediando las alturas de los puntos cercanos.
    Las unidades de la grilla son metros.

    Parámetros:
        points (np.ndarray): Nube de puntos 3D con forma (N, 3).
        grid_size (tuple): Número de divisiones en cada eje para la grilla (x, y).
        origin (tuple): Coordenadas (x_min, y_min) del origen de la grilla (en metros).
        max_coords (tuple): Coordenadas (x_max, y_max) de los límites máximos de la grilla (en metros).
        radius (float): Radio de búsqueda para encontrar puntos vecinos (en metros).
        power (float): Exponente para la ponderación por distancia.

    Retorna:
        np.ndarray: Grilla 2D de alturas (en metros).
    """
    # Crear árbol KD para búsqueda eficiente de vecinos
    tree = cKDTree(points[:, :2])

    # Crear la grilla de coordenadas en metros
    x = np.linspace(origin[0], max_coords[0], grid_size[0])
    y = np.linspace(origin[1], max_coords[1], grid_size[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')  # indexing='ij' para que eje 0 sea x y eje 1 sea y
    grid_coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Inicializar la grilla de alturas
    zz = np.full(xx.shape, np.nan)

    # Para cada celda de la grilla, calcular la altura promedio ponderada
    for idx, coord in enumerate(grid_coords):
        # Encontrar puntos dentro del radio de búsqueda
        indices = tree.query_ball_point(coord, radius)
        if indices:
            # Obtener las coordenadas y alturas de los puntos vecinos
            neighbors = points[indices]
            distances = np.linalg.norm(neighbors[:, :2] - coord, axis=1)
            # Evitar división por cero
            weights = 1 / (np.maximum(distances, 1e-6) ** power)
            weighted_avg_height = np.average(neighbors[:, 2], weights=weights)
            zz.ravel()[idx] = weighted_avg_height


    # Debug: imprime estadísticas generales de la grilla resultante
    print(f"Grid height stats: min={np.nanmin(zz):.3f}, max={np.nanmax(zz):.3f}, mean={np.nanmean(zz):.3f}")

    return zz

