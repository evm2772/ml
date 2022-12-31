
import matplotlib
matplotlib.use("TkAgg")  # Do this before importing pyplot!
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pytorch3d.utils import ico_sphere, torus
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import os
import torch
# Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
sphere_mesh = torus(R=6, r=3, rings=10, sides=10)
sphere_mesh = ico_sphere(level=0)
verts, faces, _ = load_obj("dolphin.obj")
test_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

# Differentiably sample 5k points from the surface of each mesh and then compute the loss.
#sample_sphere = sample_points_from_meshes(sphere_mesh, 5000)
sample_test = sample_points_from_meshes(test_mesh, 5000)
#loss_chamfer, _ = chamfer_distance(sample_sphere, sample_test)
print (sample_test)

ten = torch.tensor()
ten[0] = [1,2]
#for a in range(10):


def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 500)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    # ax = Axes3D(fig)
    # ax.scatter3D(x, z, -y)
    # ax.set_xlabel('x')
    # ax.set_ylabel('z')
    # ax.set_zlabel('y')
    # ax.set_title(title)
    # ax.view_init(190, 30)
    #
    # #fig, ax = plt.subplots()
    ax = fig.add_subplot(projection='3d')
    plt.xlim(min(x)-0.01, max(x)+0.01)
    plt.ylim(min(y) - 0.01, max(y) + 0.01)
    #plt.ylim(-3, 3)
    #ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(x,y,z, s=0.3, c="red")

    plt.show()


plot_pointcloud(test_mesh)