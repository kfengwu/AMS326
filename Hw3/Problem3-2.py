import numpy as np
import matplotlib.pyplot as plt

def in_rose(x, y):
    return ((x**2 + y**2)**3 <= 4 * x**2 * y**2)

def generate_cutter_points(center, width, height, angle, n=3600):
    x0, y0 = center
    theta = np.deg2rad(angle)
    
    # Uniform random points in rectangle
    u = np.random.uniform(-width/2, width/2, n)
    v = np.random.uniform(-height/2, height/2, n)
    
    # Rotate and translate
    x = x0 + u * np.cos(theta) - v * np.sin(theta)
    y = y0 + u * np.sin(theta) + v * np.cos(theta)
    
    return x, y

def estimate_overlap(center, width, height, angle):
    x, y = generate_cutter_points(center, width, height, angle)
    return np.mean(in_rose(x, y))

# Grid search
xs = np.linspace(-1, 1, 60)
ys = np.linspace(-1, 1, 60)
angles = np.linspace(0, 180, 18)  # 18 angles from 0° to 180°

best_overlap = 0
best_params = None

# Search for best (x, y, angle)
for angle in angles:
    for i, x0 in enumerate(xs):
        for j, y0 in enumerate(ys):
            overlap = estimate_overlap((x0, y0), 1, 1/np.sqrt(2), angle)
            if overlap > best_overlap:
                best_overlap = overlap
                best_params = (x0, y0, angle)

print(f"Best cutter position:")
print(f"Center: ({best_params[0]:.3f}, {best_params[1]:.3f})")
print(f"Angle: {best_params[2]:.1f}")
print(f"Estimated overlap: {best_overlap:.4f}")


# Unpack best cutter placement
best_x, best_y, best_angle = best_params

plt.figure(figsize=(8, 6))
plt.contourf(xs, ys, np.array([[estimate_overlap((x, y), 1, 1/np.sqrt(2), best_angle) 
                                for x in xs] for y in ys]), cmap='inferno')
plt.colorbar(label='Estimated Area Cut')

# Rose outline
theta_vals = np.linspace(0, 2 * np.pi, 1000)
r_vals = np.sin(2 * theta_vals)
x_vals = r_vals * np.cos(theta_vals)
y_vals = r_vals * np.sin(theta_vals)
plt.plot(x_vals, y_vals, color='blue', linewidth=1.0, label='Rose outline')

# Mark the best center
plt.plot(best_x, best_y, 'wo', markersize=8, markeredgecolor='black', label='Best Cutter Center')

# Optional: Draw the cutter rectangle outline
def draw_cutter(ax, center, width, height, angle_deg, color='black'):
    x0, y0 = center
    angle = np.deg2rad(angle_deg)
    # Rectangle corners before rotation
    corners = np.array([
        [-width/2, -height/2],
        [ width/2, -height/2],
        [ width/2,  height/2],
        [-width/2,  height/2],
        [-width/2, -height/2]  # Close loop
    ])
    # Rotation matrix
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    rotated = corners @ R.T + np.array([x0, y0])
    ax.plot(rotated[:, 0], rotated[:, 1], color=color, linestyle='-', linewidth=2, label='Best Cutter')

# Draw cutter rectangle
ax = plt.gca()
draw_cutter(ax, (best_x, best_y), 1, 1/np.sqrt(2), best_angle)

plt.xlabel("x (center of cutter)")
plt.ylabel("y (center of cutter)")
plt.title(f"Best Cutter Placement (angle = {best_angle:.1f}°)")
plt.legend(loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.show()

