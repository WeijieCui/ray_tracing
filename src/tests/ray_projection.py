from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import math


class Vector:
    # A simple 3D vector class
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, v):
        return Vector(self.x - v.x, self.y - v.y, self.z - v.z)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z


class Ray:
    # A simple ray class
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction


class Shape3D:
    # A simple 3D shape interface
    def __init__(self):
        self.center = None
        self.radius = None
        self.color = None

    # Returns the distance from the ray origin to the intersection point
    def intersect(self, ray: Ray) -> Optional[float]:
        raise NotImplementedError()


class Sphere(Shape3D):
    # A simple 3D sphere class
    def __init__(self, center, radius, color):
        # A sphere is defined by its center, radius, and color
        super().__init__()
        self.center = center
        self.radius = radius
        self.color = color

    def intersect(self, ray: Ray) -> Optional[float]:
        # Find the intersection point of the ray and the sphere
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        # If the discriminant is negative, there are no real roots
        if discriminant < 0:
            return None
        else:
            return (-b - math.sqrt(discriminant)) / (2.0 * a)


def trace(ray, objects: [Shape3D]) -> Optional[Shape3D]:
    # Find the closest sphere that the ray intersects
    closest_dist = float('inf')
    closest_sphere = None
    for sphere in objects:
        dist = sphere.intersect(ray)
        if dist is not None:
            # Check if the intersection point is in front of the camera
            if 0 <= dist < closest_dist:
                closest_dist = dist
                closest_sphere = sphere
    return closest_sphere


class Scene:
    # A simple 3D scene class
    # Contains a list of 3D objects, a camera position, size, and a background color
    def __init__(self,
                 objects: [Shape3D],
                 camera: Vector = None,
                 width: int = 100, height: int = 100, depth: int = -100,
                 ambient_light: int = 0.3,
                 background_color=None):
        if objects is None:
            objects = []
        if background_color is None:
            background_color = [ambient_light, ambient_light, ambient_light]
        if camera is None:
            camera = Vector(0, 0, 0)
        self.w = width
        self.h = height
        self.d = depth
        self.ambient_light = ambient_light
        self.camera = camera
        self.objects = objects
        self.background_color = background_color
        self.light_sources = []


def render(s: Scene):
    # Render the scene
    if s is None:
        s = Scene([])
    image = np.ones((s.h, s.w, 3))  # Initialize an empty image
    image *= s.background_color  # Set the background color
    for y in range(s.h):
        for x in range(s.w):
            direction = Vector(x - s.w / 2, y - s.h / 2, s.d)
            ray = Ray(s.camera, direction)
            sphere = trace(ray, s.objects)
            if sphere is not None:
                image[y, x] = sphere.color
                if not (s.camera.z <= sphere.center.z <= s.d or s.camera.z >= sphere.center.z >= s.d):
                    image[y, x] *= 0.5
    return image


def test_render():
    # Test the render function
    # Create a scene with multiple spheres
    objects = [
        Sphere(Vector(0, -10, -60), 12, [0, 0, 1]),
        Sphere(Vector(-10, 0, -80), 10, [1, 1, 0]),
        Sphere(Vector(0, 20, -100), 10, [0, 1, 0]),
        Sphere(Vector(20, 0, -120), 10, [1, .5, 0]),
        Sphere(Vector(10, 0, -130), 10, [1, 0, 1]),
        Sphere(Vector(0, 10, -140), 10, [0, 1, 1]),
    ]
    background_color = [.7, .5, .5]
    scene = Scene(objects=objects, background_color=background_color)
    # Render the scene from multiple camera positions
    positions = [
        [0, 0, 0],
        [5, 5, -20],
        [10, 10, -40],
        [15, 15, -40],
        [20, 20, -40],
        [25, 25, -40],
        [25, 25, -60],
        [20, 20, -80],
        [10, 10, -100],
    ]
    images = []
    for i, position in enumerate(positions):
        scene.camera = Vector(*position)
        images.append(render(scene))
    # Create a subplot for each camera position
    fig = plt.figure(figsize=(50, 50))
    for i, image in enumerate(images):
        fig.add_subplot(3, math.ceil(len(positions) / 3), i + 1)
        plt.imshow(image, origin='lower')  # Display the image
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    test_render()
