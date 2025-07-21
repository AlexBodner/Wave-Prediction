import matplotlib.pyplot as plt

def visualize_points(points, coord_system):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')


    # Plot a subsample if points are too many
    subsample = points[::max(1, len(points)//10000)]


    ax.scatter(subsample[:, 0], subsample[:, 1], subsample[:, 2],
                s=0.5, c=subsample[:, 2], cmap='jet', marker='.')

    plt.axis('equal')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Pointcloud in {coord_system} Coordinates')
    plt.show()