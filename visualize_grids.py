import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scripts_viejos_para_ver_bag.read_compressed_depths import read_compressed_image_from_bag, find_closest_image_by_timestamp
bag_path = Path('datasets/mesa_desde_lejos')
def show_closest_color_image(bag_path, target_timestamp):
    color_images = read_compressed_image_from_bag(bag_path, '/camera/camera/depth/image_rect_raw/compressedDepth')
    if not color_images:
        print("No color images found in bag.")
        return
    closest = find_closest_image_by_timestamp(color_images, target_timestamp)
    plt.figure(figsize=(10, 6))
    
    depth_meters = closest[1].astype(np.float32) / 1000.0
    depth_meters[closest[1]== 0] = np.nan 

    heatmap = plt.imshow(depth_meters, cmap='jet')
    cbar = plt.colorbar(heatmap, shrink=0.8)
    cbar.set_label('Distance (meters)', rotation=270, labelpad=15)
    
    # Remove axes ticks
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_grid(grid, title="Grid", mean_plane=None, origin=None, max_coords=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    nx, ny = grid.shape[:2]
    # Si se proveen origin y max_coords, usamos coordenadas físicas
    if origin is not None and max_coords is not None:
        x = np.linspace(origin[0], max_coords[0], nx)
        y = np.linspace(origin[1], max_coords[1], ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
    else:
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    Z = grid
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')

    # Mostrar plano medio solo donde hay datos válidos
    if mean_plane is not None:
        print("mean_plane keys:", list(mean_plane.keys()))
        if 'normal' in mean_plane and 'centroid' in mean_plane:
            normal = mean_plane['normal']
            point = mean_plane['centroid']
            if normal[2] != 0:
                # Calcular plano medio solo en la región válida
                valid_mask = np.isfinite(Z)
                X_valid = X[valid_mask]
                Y_valid = Y[valid_mask]
                Z_plane_valid = ( -normal[0]*(X_valid-point[0]) - normal[1]*(Y_valid-point[1]) ) / normal[2] + point[2]
                # Graficar el plano medio solo en los puntos válidos
                ax.scatter(X_valid, Y_valid, Z_plane_valid, color='r', alpha=0.2, s=1)
        else:
            print("mean_plane does not contain 'normal' and/or 'centroid' keys.")
    plt.show()

def main(folder):
    mean_plane_path = os.path.join(folder, "mean_plane.npz")
    mean_plane = np.load(mean_plane_path)
    for k in mean_plane.keys():
        print(f"{k}: {mean_plane[k].shape}")

    # Define origin y max_coords según cómo generaste la grilla
    # Por ejemplo, si los guardaste en mean_plane o en otro archivo, cámbialo aquí:
    origin = mean_plane['mins'] if 'mins' in mean_plane else None
    max_coords = mean_plane['maxs'] if 'maxs' in mean_plane else None
    nu, nv = 50, 50  # las dim de la grilla
    spacing = 0.02  # mismo que usaste para muestreo

    # Visualizar todas las grillas
    for fname in os.listdir(folder):
        if fname.startswith("grid_") and fname.endswith(".npy"):
            grid = np.load(os.path.join(folder, fname))

            plot_grid(grid, title=fname, mean_plane=None, origin=origin, max_coords=max_coords)
            timestamp_str = fname.split('_')[1].split('.')[0]
            try:
                timestamp = int(timestamp_str)
            except ValueError:
                print(f"Could not parse timestamp from {fname}")
                continue
            show_closest_color_image(bag_path, timestamp)

if __name__ == "__main__":
    folder = "numpy_grids"
    main(folder)


def plane_uv_grid_to_world(nu, nv, centroid, u_vec, v_vec, spacing=0.02):
    """
    Genera una grilla (nu x nv) de coordenadas (u,v) proyectadas al sistema XYZ mundo.

    Args:
        nu, nv: dimensiones de la grilla
        centroid: punto en el plano (XYZ)
        u_vec, v_vec: vectores unitarios en el plano
        spacing: distancia entre puntos en metros

    Returns:
        world_coords: ndarray (nu, nv, 3) con las coordenadas XYZ
    """
    us = (np.arange(nu) - (nu-1)/2) * spacing
    vs = (np.arange(nv) - (nv-1)/2) * spacing

    world_coords = np.zeros((nu, nv, 3))
    for i, ui in enumerate(us):
        for j, vj in enumerate(vs):
            world_coords[i, j] = centroid + ui * u_vec + vj * v_vec

    return world_coords