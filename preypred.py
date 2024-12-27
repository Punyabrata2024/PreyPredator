import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
N = 100  # Number of prey
v0 = 0.3  # Speed of each prey
R = 0.5  # Prey interaction radius
r = 0.01  # Predator kill radius
alpha = 1
beta = 1
gamma = 0.2
delta = 2.5
meu = 1
meu_pred = 1
L = 0.5  # Scale for simulation area
dt = 0.01  # Time step
frame_margin = 0.5  # Margin around predator for dynamic frame adjustment

# Initialize prey positions
positions = L * (2 * np.random.rand(N, 2) - 1)  # Random positions in a square

# Initialize predator
predator_pos = np.array([L, 0.0])
predator_vel = np.zeros(2)

# Initialize velocities as zeros for prey
velocities = np.zeros_like(positions)

# Helper function to calculate distances
def compute_distances(p1, p2):
    return np.linalg.norm(p1[:, None] - p2, axis=2)

# Force calculation for prey
def compute_prey_forces(positions, predator_pos):
    forces = np.zeros_like(positions)

    for i in range(N):
        # Compute distances to all other prey
        distances = np.linalg.norm(positions - positions[i], axis=1)
        neighbors = (distances < R) & (distances > 0)

        # Compute neighbor influence
        if np.any(neighbors):
            neighbor_vecs = positions[neighbors] - positions[i]
            neighbor_dists = distances[neighbors][:, None]
            neighbor_forces = beta * neighbor_vecs - alpha * neighbor_vecs / (neighbor_dists**2)
            forces[i] += np.sum(neighbor_forces, axis=0) / len(neighbor_vecs)

        # Add predator influence
        predator_vec = predator_pos - positions[i]
        predator_dist = np.linalg.norm(predator_vec)
        if predator_dist > 0:
            forces[i] -= gamma * predator_vec / (predator_dist**2)

    return forces

# Force calculation for predator
def compute_predator_force(positions, predator_pos):
    dists = positions - predator_pos
    magnitudes = np.linalg.norm(dists, axis=1)
    valid = magnitudes > 0
    forces = dists[valid] / magnitudes[valid][:, None] ** 3
    return gamma * np.sum(forces, axis=0) / len(positions)

# Update positions and velocities
def update_positions(positions, velocities, predator_pos, predator_vel):
    global N
    # Compute forces
    prey_forces = compute_prey_forces(positions, predator_pos)
    predator_force = compute_predator_force(positions, predator_pos)

    # Update predator position
    predator_vel += (1 / meu_pred) * predator_force * dt
    predator_pos += predator_vel * dt

    # Update prey positions
    velocities += (1 / meu) * prey_forces * dt
    positions += velocities * dt

    # Handle prey removal near predator
    distances_to_predator = np.linalg.norm(positions - predator_pos, axis=1)
    valid_indices = distances_to_predator >= r
    positions = positions[valid_indices]
    velocities = velocities[valid_indices]
    N = len(positions)  # Update number of prey dynamically

    return positions, velocities, predator_pos, predator_vel

# Set up Matplotlib animation
fig, ax = plt.subplots()

# Initialize frame limits
frame_limits = {
    "x_min": -L - frame_margin,
    "x_max": L + frame_margin,
    "y_min": -L - frame_margin,
    "y_max": L + frame_margin,
}

# Initial plot setup
prey_scatter = ax.scatter(positions[:, 0], positions[:, 1], color='blue', label='Prey')
predator_scatter = ax.scatter(predator_pos[0], predator_pos[1], color='red', label='Predator')

def animate(frame):
    global positions, velocities, predator_pos, predator_vel
    positions, velocities, predator_pos, predator_vel = update_positions(
        positions, velocities, predator_pos, predator_vel
    )

    # Update prey and predator positions
    prey_scatter.set_offsets(positions)
    predator_scatter.set_offsets(predator_pos)

    # Dynamically adjust frame limits based on predator position
    frame_limits["x_min"] = min(frame_limits["x_min"], predator_pos[0] - frame_margin)
    frame_limits["x_max"] = max(frame_limits["x_max"], predator_pos[0] + frame_margin)
    frame_limits["y_min"] = min(frame_limits["y_min"], predator_pos[1] - frame_margin)
    frame_limits["y_max"] = max(frame_limits["y_max"], predator_pos[1] + frame_margin)

    # Update axis limits
    ax.set_xlim(frame_limits["x_min"], frame_limits["x_max"])
    ax.set_ylim(frame_limits["y_min"], frame_limits["y_max"])

    ax.legend()
    return prey_scatter, predator_scatter

ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
plt.show()
