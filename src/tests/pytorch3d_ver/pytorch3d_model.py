import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
import matplotlib.pyplot as plt

# Define the box corners and faces
verts = torch.tensor([
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1],
], dtype=torch.float32)

faces = torch.tensor([
    [0, 2, 1],
    [1, 2, 3],
    [2, 6, 3],
    [3, 6, 7],
    [0, 4, 2],
    [2, 4, 6],
    [0, 1, 4],
    [1, 5, 4],
    [4, 5, 6],
    [5, 7, 6],
    [1, 3, 5],
    [3, 7, 5],
], dtype=torch.int64)

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb)

# Create a Meshes object for the box.
mesh = Meshes(
    verts=[verts],
    faces=[faces],
    textures=textures
)

# Initialize a camera.
R, T = look_at_view_transform(2.7, 0, 0)
cameras = FoVPerspectiveCameras(device="cpu", R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0.
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Place a point light in front of the object.
lights = PointLights(device="cpu", location=[[0.0, 0.0, -3.0]])

# Create a phong renderer by composing a rasterizer and a shader.
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device="cpu",
        cameras=cameras,
        lights=lights
    )
)

# Render the box
images = renderer(mesh)

# Visualize the image
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].detach().cpu().numpy())
plt.axis("off")
plt.show()