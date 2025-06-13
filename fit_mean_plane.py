import numpy as np
import pyransac3d as pyrsc
import pyvista as pv
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
def fit_plane_ransac(points, thresh=0.1, max_iterations=100):
    """
    Ajusta un plano usando RANSAC.
    Devuelve:
      - centroid: un punto en el plano
      - normal: vector unidad perpendicular
      - u, v: ejes ortogonales en el plano
      - inliers: indices de puntos inliers
    """
    plane = pyrsc.Plane()
    best_eq, inliers = plane.fit(points, thresh, max_iterations)
    A, B, C, D = best_eq
    normal = np.array([A, B, C])
    normal /= np.linalg.norm(normal)

    # Una solución de un punto cualquiera en el plano:
    centroid = -D * normal

    # Ejes en el plano:
    u = np.cross(normal, [0, 0, 1])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(normal, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    return centroid, normal, u, v, inliers

def sample_on_plane(mesh, centroid, u, v, normal, grid_size=(256,256), spacing=0.02):
    """
    Muestrea la malla en una grilla equiespaciada en el plano definido por centro, u, v.
    Retorna:
      - points_world: coordenadas XYZ reales (shape nu×nv×3)
      - uvz: coordenadas (u, v, z_local) en sistema del plano
    """
    nu, nv = grid_size
    us = (np.arange(nu) - (nu-1)/2) * spacing
    vs = (np.arange(nv) - (nv-1)/2) * spacing
    pts_world = np.zeros((nu, nv, 3))
    uvz = np.zeros((nu, nv, 3))
    
    kdt = cKDTree(mesh.points)

    for i, ui in enumerate(us):
        for j, vj in enumerate(vs):
            pw = centroid + ui*u + vj*v
            _, idx = kdt.query(pw)
            pt_mesh = mesh.points[idx]
            z_local = np.dot(pt_mesh - centroid, normal)
            pts_world[i, j] = pt_mesh
            uvz[i, j] = [ui, vj, z_local]

    return pts_world, uvz

def make_structured_grid(points_world, uvz):
    """
    Convierte los resultados en un StructuredGrid de PyVista para visualización.
    """
    nu, nv, _ = points_world.shape
    grid = pv.StructuredGrid()
    grid.points = points_world.reshape(-1, 3)
    grid.dimensions = [nv, nu, 1]
    grid["z_local"] = uvz[:, :, 2].ravel()
    return grid

def process_and_plot(points, mesh=None, thresh=0.01, max_iterations=1000,
                     grid_size=(50,50), spacing=0.02):
    """
    Flujo completo:
      1. Fittea plano por RANSAC
      2. Crea o toma la mesh
      3. Muestra en grilla equiespaciada proyectada al plano
      4. Visualiza con PyVista
    """
    centroid, normal, u, v, inliers = fit_plane_ransac(points, thresh, max_iterations)
    mesh = mesh or pv.PolyData(points).delaunay_2d()
    pts_world, uvz = sample_on_plane(mesh, centroid, u, v, normal, grid_size, spacing)
    grid = make_structured_grid(pts_world, uvz)

    p = pv.Plotter()
    p.add_mesh(mesh, color="lightgray", opacity=0.5, label="mesh")
    p.add_mesh(grid, scalars="z_local", cmap="viridis", show_scalar_bar=True, label="plane-sample")
    p.add_legend()
    p.show()

    return {
        "centroid": centroid,
        "normal": normal,
        "u": u,
        "v": v,
        "inliers": inliers,
        "mesh": mesh,
        "structured_grid": grid,
        "z_local": uvz[:, :, 2]
    }
def plot_plane_with_points(points, centroid, normal, scale=1.0):
    """
    Grafica una nube de puntos y el plano ajustado.

    points: np.ndarray (N,3)
    centroid: punto en el plano
    normal: vector normal unitario
    scale: extensión del plano con respecto al spread de puntos
    """
    # Muestra la nube de puntos (submuestra si es necesario)
    subs = points[::max(1, len(points)//10000)]
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(subs[:,0], subs[:,1], subs[:,2],
               s=1, c='b', alpha=0.5, label='Nube de puntos')

    # Definir dos ejes u y v en el plano
    # Encuentra cualquier vector no paralelo a normal
    not_parallel = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, not_parallel)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Crear meshgrid en el plano
    d = scale * np.max(np.ptp(points, axis=0))
    uu = np.linspace(-d, d, 10)
    vv = np.linspace(-d, d, 10)
    U, V = np.meshgrid(uu, vv)
    plane_pts = centroid + U[...,None]*u + V[...,None]*v
    Xp, Yp, Zp = plane_pts[...,0], plane_pts[...,1], plane_pts[...,2]
    ax._facecolors2d = ax._facecolor
    #ax._edgecolors2d = ax._edgecolor 
    ax.plot_surface(Xp, Yp, Zp, color='r', alpha=0.3, label='Plano fitteado')

    # Etiquetas y vista
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Nube de puntos  y plano ajustado por RANSAC')
    #plt.legend()
    plt.show()
