import numpy as np
from scipy.spatial import KDTree, Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def coordinate_model(x: float = 436463.00052670995,
                     y: float = 3547037.885359012,
                     z: float = 5265322.612163001,
                     amount: int = 10,
                     min_dist: float = 10000,
                     max_dist: float = 20000,
                     coordinate_system: str = 'ecef'):
    """
    Generates a network of points with random coordinates

    Args:
        x (float): Initial coordinate X
        y (float): Initial coordinate Y
        z (float): Initial coordinate Z
        amount (int): Number of points to generate
        min_dist (float): Minimum distance between points
        max_dist (float): Maximum distance between points

    Returns:
        list: List of points with coordinates
    """
    if min_dist > max_dist:
        raise ValueError("Minimum distance cannot be greater than maximum distance")

    points = [{'Station': ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 4)), 'X': x, 'Y': y, 'Z': z}]

    while len(points) < amount:
        direction = np.random.uniform(0, 2 * np.pi)  # Generate random direction
        distance = np.random.uniform(min_dist, max_dist)  # Generate random distance within range
        prev_x, prev_y, prev_z = points[-1]['X'], points[-1]['Y'], points[-1]['Z']  # get previous point coordinates

        # Calculate new point's coordinates
        x = prev_x + distance * np.cos(direction)
        y = prev_y + distance * np.sin(direction)
        z = prev_z + np.random.uniform(-25, 25)

        station_name = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 4))
        point = {'Station': station_name, 'X': x, 'Y': y, 'Z': z, }

        tree = KDTree([[p['X'], p['Y'], p['Z']] for p in points])

        if len(points) < 2:
            neighbours = 1
        else:
            neighbours = 2

        dist, _ = tree.query((x, y, z), k=neighbours)
        if np.all(min_dist <= dist) and np.all(dist <= max_dist):
            points.append(point)

    return points


points = coordinate_model()
print(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = [point['X'] for point in points]
ys = [point['Y'] for point in points]
zs = [point['Z'] for point in points]

ax.scatter(xs, ys, zs)

for i, point in enumerate(points):
    ax.text(point['X'], point['Y'], point['Z'], point['Station'])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
