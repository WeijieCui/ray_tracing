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

    def __mul__(self, v):
        return Vector(self.x * v, self.y * v, self.z * v)

    def __add__(self, v):
        if isinstance(v, Vector):
            return Vector(self.x + v.x, self.y + v.y, self.z + v.z)
        else:
            return Vector(self.x + v, self.y + v, self.z + v)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def normalize(self):
        length = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        if length == 0:
            return Vector(0, 0, 0)
        return Vector(self.x / length, self.y / length, self.z / length)


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
    def intersect(self, ray: Ray, scene) -> (Optional[float], Optional[Vector]):
        raise NotImplementedError()


class Light:
    def __init__(self, position: Vector, intensity: float):
        self.position = position
        self.intensity = intensity


class Scene:
    # A simple 3D scene class
    # Contains a list of 3D objects, a camera position, size, and a background color
    def __init__(self,
                 objects: [Shape3D],
                 lights: [Light],
                 camera: Vector = None,
                 w: int = 100, h: int = 100, d: int = -100,
                 ambient_light: int = 0.3,
                 background_color=None):
        self.objects = objects if objects is not None else []
        self.lights = lights if lights is not None else []
        self.camera = camera if camera is not None else Vector(0, 0, 0)
        self.w = w
        self.h = h
        self.d = d
        self.ambient_light = ambient_light
        self.background_color = background_color if background_color is not None else [ambient_light] * 3
        self.light_sources = []


class Sphere(Shape3D):
    # A simple 3D sphere class
    def __init__(self, center, radius, color):
        # A sphere is defined by its center, radius, and color
        super().__init__()
        self.center = center
        self.radius = radius
        self.color = color

    def intersect(self, ray: Ray, scene: Scene) -> (Optional[float], Optional[Vector]):
        # Find the intersection point of the ray and the sphere
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        # If the discriminant is negative, there are no real roots
        if discriminant < 0:
            return None, None
        else:
            t = (-b - math.sqrt(discriminant)) / (2.0 * a)
            intersection_point = ray.origin + ray.direction * t
            normal = (intersection_point - self.center).normalize()
            color = [i * scene.ambient_light for i in self.color]
            for light in scene.lights:
                light_dir = (light.position - intersection_point).normalize()
                for i in range(len(color)):
                    color[i] += light.intensity * self.color[i] * normal.dot(light_dir)
            return t, color


class Cube(Shape3D):
    def __init__(self, min_corner, max_corner, color):
        super().__init__()
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.color = color

    def intersect(self, ray: Ray, scene: Scene) -> (Optional[float], Optional[Vector]):
        # Compute the intersection of the ray with the cube
        tmin = 0 if ray.direction.x == 0 else (self.min_corner.x - ray.origin.x) / ray.direction.x
        tmax = 0 if ray.direction.x == 0 else (self.max_corner.x - ray.origin.x) / ray.direction.x

        if tmin > tmax:
            tmin, tmax = tmax, tmin

        tymin = 0 if ray.direction.y == 0 else (self.min_corner.y - ray.origin.y) / ray.direction.y
        tymax = 0 if ray.direction.y == 0 else (self.max_corner.y - ray.origin.y) / ray.direction.y

        if tymin > tymax:
            tymin, tymax = tymax, tymin

        if (tmin > tymax) or (tymin > tmax):
            return None, None

        if tymin > tmin:
            tmin = tymin

        if tymax < tmax:
            tmax = tymax

        tzmin = 0 if ray.direction.z == 0 else (self.min_corner.z - ray.origin.z) / ray.direction.z
        tzmax = 0 if ray.direction.z == 0 else (self.max_corner.z - ray.origin.z) / ray.direction.z

        if tzmin > tzmax:
            tzmin, tzmax = tzmax, tzmin

        if (tmin > tzmax) or (tzmin > tmax):
            return None, None

        if tzmin > tmin:
            tmin = tzmin

        if tzmax < tmax:
            tmax = tzmax

        intersection_point = ray.origin + ray.direction * tmin
        normal = (intersection_point - self.min_corner).normalize()
        color = [i * scene.ambient_light for i in self.color]
        for light in scene.lights:
            light_dir = (light.position - intersection_point).normalize()
            for i in range(len(color)):
                color[i] += light.intensity * self.color[i] * normal.dot(light_dir)
        return tmin, color


def trace(ray, scene: Scene) -> Optional[Shape3D]:
    # Find the closest sphere that the ray intersects
    closest_dist = float('inf')
    closest_color = None
    for shape3d in scene.objects:
        dist, color = shape3d.intersect(ray, scene)
        if dist is not None:
            # Check if the intersection point is in front of the camera
            if 0 <= dist < closest_dist:
                closest_dist = dist
                closest_color = color
    return closest_color


def render(s: Scene):
    # Render the scene
    image = np.ones((s.h, s.w, 3))  # Initialize an empty image
    image *= s.background_color  # Set the background color
    for y in range(s.h):
        for x in range(s.w):
            direction = Vector(x - s.w / 2, y - s.h / 2, s.d)
            ray = Ray(s.camera, direction)
            color = trace(ray, s)
            if color is not None:
                image[y, x] = color
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
        Cube(Vector(-100, -100, -150), Vector(100, 100, -151), [0.5, 0.5, 0]),
        Cube(Vector(-35, -35, -150), Vector(35, -35, 0), [0.5, 0.5, 0]),
        Cube(Vector(-35, 35, -150), Vector(35, 35, 0), [0.5, 0.5, 0]),
        Cube(Vector(-35, -35, -150), Vector(-35, 35, 0), [0.5, 0.5, 0]),
        Cube(Vector(35, 35, -150), Vector(35, -35, 0), [0.5, 0.5, 0]),
        Cube(Vector(30, 35, -80), Vector(25, 30, -100), [0.5, 0.5, 0]),
        Cube(Vector(30, -35, -80), Vector(25, -25, -100), [0.5, 0.5, 0]),
    ]
    lights = [
        Light(Vector(-30, -30, -0), 0.2),
        Light(Vector(30, 30, -50), 0.5),
        Light(Vector(15, -10, -50), 0.8),
    ]
    background_color = [.7, .5, .5]
    scene = Scene(
        objects=objects,
        lights=lights,
        background_color=background_color
    )
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
    fig = plt.figure(figsize=(60, 60))
    for i, image in enumerate(images):
        fig.add_subplot(3, math.ceil(len(positions) / 3), i + 1)
        plt.imshow(image, origin='lower')  # Display the image
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    test_render()
