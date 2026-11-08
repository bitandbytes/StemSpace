"""Perspective projection for the static placement editor."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float

    def __add__(self, other: Point3) -> Point3:
        return Point3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Point3) -> Point3:
        return Point3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: float) -> Point3:
        return Point3(self.x * scale, self.y * scale, self.z * scale)

    def dot(self, other: Point3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z


ORIGIN = Point3(0, 0, 0)
# X is listener-right, Y is listener-front, and Z is up. Radius is for display
# only: the placement model has direction and width, with no distance control.
CAMERA_TARGET = Point3(0, 0, 0.28)
CAMERA_DISTANCE = 4.2


def wrap_azimuth(angle: float) -> float:
    return (angle + 180) % 360 - 180


def direction(azimuth: float, elevation: float, radius: float = 1.0) -> Point3:
    az, el = radians(azimuth), radians(elevation)
    if elevation == 90:
        return Point3(0, 0, radius)
    return Point3(radius * sin(az) * cos(el), radius * cos(az) * cos(el), radius * sin(el))


def azimuth_of(point: Point3) -> float:
    return wrap_azimuth(degrees(atan2(point.x, point.y)))


def width_arc(azimuth: float, elevation: float, width: float,
              radius: float = 1.0, steps: int = 20) -> list[Point3]:
    start = azimuth - width / 2
    return [direction(start + width * i / steps, elevation, radius) for i in range(steps + 1)]


def gain_radius(gain_db: float) -> float:
    """Screen-space visual gain scale. The center hit target does not scale."""
    return 8 + 7 * 10 ** (gain_db / 40) if gain_db <= 0 else 15 + 0.8 * gain_db


@dataclass(frozen=True)
class Camera:
    yaw: float = 32
    pitch: float = 49
    zoom: float = 1

    def orbit(self, dx: float, dy: float) -> Camera:
        return Camera(wrap_azimuth(self.yaw - dx * 0.45),
                      min(82, max(14, self.pitch + dy * 0.36)), self.zoom)

    def dolly(self, steps: float) -> Camera:
        return Camera(self.yaw, self.pitch, min(1.75, max(0.60, self.zoom * 1.15 ** steps)))


@dataclass(frozen=True)
class Projected:
    x: float
    y: float
    depth: float


class Projection:
    def __init__(self, camera: Camera, width: int, height: int):
        yaw, pitch = radians(camera.yaw), radians(camera.pitch)
        self.right = Point3(cos(yaw), sin(yaw), 0)
        self.up = Point3(-sin(yaw) * sin(pitch), cos(yaw) * sin(pitch), cos(pitch))
        self.normal = Point3(sin(yaw) * cos(pitch), -cos(yaw) * cos(pitch), sin(pitch))
        self.eye = CAMERA_TARGET + self.normal * CAMERA_DISTANCE
        self.focal = max(1, min(width, height)) * 1.62 * camera.zoom
        self.cx = width / 2
        self.cy = height * 0.52

    def project(self, point: Point3) -> Projected:
        offset = point - self.eye
        depth = -offset.dot(self.normal)
        return Projected(self.cx + self.focal * offset.dot(self.right) / depth,
                         self.cy - self.focal * offset.dot(self.up) / depth, depth)

    def on_horizontal(self, x: float, y: float, z: float) -> Point3 | None:
        """Intersect a screen ray with a constant-height plane."""
        ray = (self.right * ((x - self.cx) / self.focal)
               + self.up * ((self.cy - y) / self.focal) - self.normal)
        if abs(ray.z) < 1e-8:
            return None
        t = (z - self.eye.z) / ray.z
        return self.eye + ray * t if t > 0 else None
