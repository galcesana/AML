import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# region: Conditional Helpers

def generate_points_on_ring(center, radius, thickness, num_points):
    points = []
    while len(points) < num_points:
        r = np.random.uniform(radius - thickness / 2, radius + thickness / 2)
        theta = np.random.uniform(0, 2 * np.pi)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        points.append((x, y))
    return points


def sample_olympic_rings(num_points_per_ring, ring_thickness=0.1):
    centers = [(0, 0), (2, 0), (4, 0), (1, -1), (3, -1)]
    colors = ['blue', 'black', 'red', 'yellow', 'green']
    radius = 1
    all_points = []
    all_labels = []

    for center, color in zip(centers, colors):
        points = generate_points_on_ring(center, radius, ring_thickness, num_points_per_ring)
        labels = [color] * num_points_per_ring
        all_points.extend(points)
        all_labels.extend(labels)

    return all_points, all_labels


# endregion: Conditional Helpers

# region: Unconditional Helpers

def point_in_ring(x, y, center, radius, thickness):
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return radius - thickness / 2 <= distance <= radius + thickness / 2


def generate_points_on_rings__unconditional(centers, radius, thickness, num_points):
    points = []
    count = 0
    while count < num_points:
        x = np.random.uniform(-1, 5)
        y = np.random.uniform(-2, 1)
        in_any_ring = False
        for center in centers:
            if point_in_ring(x, y, center, radius, thickness):
                in_any_ring = True
                break
        if in_any_ring:
            points.append((x, y))
            count += 1
    return points


# endregion: Unconditional Helpers


def create_olympic_rings(n_points, ring_thickness=0.25, verbose=True):
    num_points_per_ring = n_points // 5
    sampled_points, labels = sample_olympic_rings(num_points_per_ring, ring_thickness)

    # Plotting the points
    if verbose:
        x, y = zip(*sampled_points)
        colors = labels
        if len(sampled_points) > 10000:
            rand_idx = np.random.choice(len(sampled_points), 10000, replace=False)
            plt.scatter(np.array(x)[rand_idx], np.array(y)[rand_idx], s=1, c=np.array(colors)[rand_idx])
        else:
            plt.scatter(x, y, s=1, c=colors)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Numpy Sampled Olympic Rings')
        plt.show()

    sampled_points = np.asarray(sampled_points)
    # transform labels from strings to ints
    label_to_int = {k: v for v, k in enumerate(np.unique(labels))}
    int_to_label = {v: k for k, v in label_to_int.items()}
    labels = np.array([label_to_int[label] for label in labels])

    sampled_points = np.asarray(sampled_points)
    # normalize data
    sampled_points = (sampled_points - np.mean(sampled_points, axis=0)) / np.std(sampled_points, axis=0)

    return sampled_points, labels, int_to_label


def create_unconditional_olympic_rings(n_points, ring_thickness=0.25, verbose=True):
    centers = [(0, 0), (2, 0), (4, 0), (1, -1), (3, -1)]
    radius = 1
    data = generate_points_on_rings__unconditional(centers, radius, ring_thickness, n_points)
    if verbose:
        x, y = zip(*data)
        if len(data) > 10000:
            rand_idx = np.random.choice(len(data), 10000, replace=False)
            plt.scatter(np.array(x)[rand_idx], np.array(y)[rand_idx], s=1)
        else:
            plt.scatter(x, y, s=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Numpy Sampled Olympic Rings')
        plt.show()
    data = np.asarray(data)
    # normalize data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return data
