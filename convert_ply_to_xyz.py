import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("mesh.ply")  # binary ply
np.savetxt("mesh_converted.xyz", np.asarray(pcd.points), fmt="%.6f")
