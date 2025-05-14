import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

def determine_params():
    height = random.random() * 40 + 150
    head_ratio = random.random() * 2 + 6
    head_length = height / head_ratio
    body_length = height - head_length
    upper_lower_ratio = random.random() * 0.03 + 0.5
    upper_length = body_length * upper_lower_ratio
    lower_length = body_length - upper_length
    body_width = 0.25 * height
    arm_length = (height - body_width) / 2
    return height, head_length, upper_length, lower_length, body_width, arm_length

def create_sphere(center, radius, resolution=20):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def create_ellipsoid(center, rx, ry, rz, resolution=20):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + rx * np.outer(np.cos(u), np.sin(v))
    y = center[1] + ry * np.outer(np.sin(u), np.sin(v))
    z = center[2] + rz * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def create_cuboid(center, width, height, depth):
    half_w, half_h, half_d = width/2, height/2, depth/2
    vertices = np.array([
        [center[0]-half_w, center[1]-half_h, center[2]-half_d],
        [center[0]+half_w, center[1]-half_h, center[2]-half_d],
        [center[0]+half_w, center[1]+half_h, center[2]-half_d],
        [center[0]-half_w, center[1]+half_h, center[2]-half_d],
        [center[0]-half_w, center[1]-half_h, center[2]+half_d],
        [center[0]+half_w, center[1]-half_h, center[2]+half_d],
        [center[0]+half_w, center[1]+half_h, center[2]+half_d],
        [center[0]-half_w, center[1]+half_h, center[2]+half_d]
    ])
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]
    return faces

def add_random_internal_structures(ax, center, max_radius, num_structures=5):
    for _ in range(num_structures):
        struct_radius = random.uniform(max_radius * 0.2, max_radius * 0.5)
        while True:
            struct_center = [
                center[0] + random.uniform(-max_radius * 0.4, max_radius * 0.4),
                center[1] + random.uniform(-max_radius * 0.4, max_radius * 0.4),
                center[2] + random.uniform(-max_radius * 0.4, max_radius * 0.4)
            ]
            distance = np.linalg.norm(np.array(struct_center) - np.array(center))
            if distance + struct_radius <= max_radius:
                break
        x, y, z = create_sphere(struct_center, struct_radius, resolution=20)
        ax.plot_surface(x, y, z, color='gray', alpha=0.9)

def generate_3d_phantom_with_random_structures():
    height, head_length, upper_length, lower_length, body_width, arm_length = determine_params()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    head_radius = head_length / 2
    head_center = [0, 0, height - head_radius]
    head_x, head_y, head_z = create_sphere(head_center, head_radius)
    ax.plot_surface(head_x, head_y, head_z, color='bisque', alpha=0.7)
    add_random_internal_structures(ax, head_center, head_radius, num_structures=3)
    torso_center = [0, 0, height - head_length - upper_length / 2]
    torso_x, torso_y, torso_z = create_ellipsoid(torso_center, body_width / 2, body_width / 4, upper_length / 2)
    ax.plot_surface(torso_x, torso_y, torso_z, color='lightblue', alpha=0.7)
    add_random_internal_structures(ax, torso_center, min(body_width / 2, upper_length / 2), num_structures=5)
    arm_width, arm_thickness = body_width * 0.15, body_width * 0.15
    arm_vertical_position = height - head_length - upper_length * 0.25
    left_arm_center = [-body_width/2 - arm_width/2, 0, arm_vertical_position - arm_length/2]
    right_arm_center = [body_width/2 + arm_width/2, 0, arm_vertical_position - arm_length/2]
    left_arm_faces = create_cuboid(left_arm_center, arm_width, arm_thickness, arm_length)
    right_arm_faces = create_cuboid(right_arm_center, arm_width, arm_thickness, arm_length)
    for faces in [left_arm_faces, right_arm_faces]:
        ax.add_collection3d(Poly3DCollection(faces, color='lightgreen', alpha=0.7))
    leg_width, leg_thickness = body_width * 0.2, body_width * 0.2
    left_leg_center = [-body_width/4, 0, height - head_length - upper_length - lower_length/2]
    right_leg_center = [body_width/4, 0, height - head_length - upper_length - lower_length/2]
    left_leg_faces = create_cuboid(left_leg_center, leg_width, leg_thickness, lower_length)
    right_leg_faces = create_cuboid(right_leg_center, leg_width, leg_thickness, lower_length)
    for faces in [left_leg_faces, right_leg_faces]:
        ax.add_collection3d(Poly3DCollection(faces, color='lightcoral', alpha=0.7))
    ax.set_xlim(-height/2, height/2)
    ax.set_ylim(-height/2, height/2)
    ax.set_zlim(0, height)
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title('3D Human Phantom with Random Internal Structures')
    plt.savefig("phantom.png")

generate_3d_phantom_with_random_structures()

