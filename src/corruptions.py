"""
Environmental corruption generators for evaluating robustness of traffic sign classification models.
Implements 14 corruptions (weather, noise, blur, digital, and geometric artifacts)
across 5 severity levels following benchmark standards.
"""

import io
import math
import random
import cv2
import numpy as np
from PIL import Image


def disk_kernel(radius: int) -> np.ndarray:
    """Generates a normalized circular disk filter kernel for defocus blur."""
    radius = max(1, int(radius))
    grid_coords = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(grid_coords, grid_coords)
    kernel = np.array((x ** 2 + y ** 2) <= radius ** 2, dtype=np.float32)
    return kernel / (np.sum(kernel) + 1e-8)


def motion_kernel(angle: float, length: int) -> np.ndarray:
    """Generates a normalized directional motion blur filter kernel."""
    length = max(1, int(length))
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    cos_val = math.cos(math.radians(angle))
    sin_val = math.sin(math.radians(angle))

    for i in range(length):
        offset = i - center
        row = int(round(center + offset * sin_val))
        col = int(round(center + offset * cos_val))
        if 0 <= row < length and 0 <= col < length:
            kernel[row, col] = 1.0

    return kernel / (np.sum(kernel) + 1e-8)


class EnvironmentalCorruptions:
    """
    Standardized suite of 14 environmental corruption operators
    applicable to normalized floating-point images in the range [0.0, 1.0].
    """

    @staticmethod
    def apply_gaussian_noise(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Applies zero-mean additive Gaussian white noise."""
        severity = max(1, min(5, int(severity)))
        c = [0.04, 0.08, 0.12, 0.18, 0.26][severity - 1]
        noise = np.random.normal(0.0, c, img_np.shape).astype(np.float32)
        return np.clip(img_np + noise, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_shot_noise(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Applies Poisson photon shot noise."""
        severity = max(1, min(5, int(severity)))
        c = [500, 250, 100, 50, 20][severity - 1]
        scaled = np.clip(img_np * c, 0.0, None)
        noisy = np.random.poisson(scaled).astype(np.float32) / float(c)
        return np.clip(noisy, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_impulse_noise(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Applies salt-and-pepper impulse noise."""
        severity = max(1, min(5, int(severity)))
        c = [0.02, 0.05, 0.09, 0.15, 0.24][severity - 1]
        out = img_np.copy()
        prob = np.random.rand(*img_np.shape[:2])
        out[prob < (c / 2.0)] = 0.0
        out[(prob >= (c / 2.0)) & (prob < c)] = 1.0
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_defocus_blur(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Applies optical defocus blur using disk convolution."""
        severity = max(1, min(5, int(severity)))
        radius = [1, 2, 3, 4, 6][severity - 1]
        kernel = disk_kernel(radius)
        blurred = cv2.filter2D(img_np, -1, kernel)
        return np.clip(blurred, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_motion_blur(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Applies directional camera or vehicle motion blur."""
        severity = max(1, min(5, int(severity)))
        length = [5, 9, 13, 17, 23][severity - 1]
        angle = random.uniform(0, 360)
        kernel = motion_kernel(angle, length)
        blurred = cv2.filter2D(img_np, -1, kernel)
        return np.clip(blurred, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_zoom_blur(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Applies radial zoom/scale blur simulating camera approach."""
        severity = max(1, min(5, int(severity)))
        zoom_factors = [
            np.linspace(1.0, 1.06, 3),
            np.linspace(1.0, 1.12, 4),
            np.linspace(1.0, 1.18, 5),
            np.linspace(1.0, 1.24, 6),
            np.linspace(1.0, 1.32, 7)
        ][severity - 1]

        h, w = img_np.shape[:2]
        accum = np.zeros_like(img_np, dtype=np.float32)
        for zoom in zoom_factors:
            nh, nw = int(h * zoom), int(w * zoom)
            resized = cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_LINEAR)
            sy, sx = (nh - h) // 2, (nw - w) // 2
            accum += resized[sy:sy + h, sx:sx + w]

        return np.clip(accum / len(zoom_factors), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_snow(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Simulates snow occlusions and winter haze."""
        severity = max(1, min(5, int(severity)))
        params = [
            (0.10, 0.20, 0.70, 0.50),
            (0.20, 0.30, 0.60, 0.60),
            (0.35, 0.40, 0.50, 0.70),
            (0.50, 0.45, 0.40, 0.80),
            (0.65, 0.50, 0.30, 0.85)
        ][severity - 1]

        loc, scale, blend_orig, blend_snow = params
        h, w = img_np.shape[:2]
        snow_raw = np.random.normal(loc=loc, scale=scale, size=(h, w)).astype(np.float32)
        snow_layer = cv2.GaussianBlur(snow_raw, (3, 3), 0)
        snow_layer = np.clip(snow_layer, 0.0, 1.0)[..., np.newaxis]

        out = img_np * blend_orig + snow_layer * blend_snow
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_frost(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Simulates ice crystals and frosty windshield patterns."""
        severity = max(1, min(5, int(severity)))
        c = [0.20, 0.35, 0.50, 0.65, 0.80][severity - 1]
        h, w, ch = img_np.shape
        frost_noise = cv2.GaussianBlur(
            np.random.uniform(0.6, 1.0, (h, w, ch)).astype(np.float32),
            (7, 7),
            2
        )
        out = (1.0 - c) * img_np + c * frost_noise
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_fog(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Simulates dense atmospheric fog and light scattering."""
        severity = max(1, min(5, int(severity)))
        c = [0.20, 0.35, 0.50, 0.65, 0.80][severity - 1]
        h, w = img_np.shape[:2]
        max_dim = max(h, w)
        t = np.linspace(-1.0, 1.0, max_dim, dtype=np.float32)
        x, y = np.meshgrid(t, t)
        dist = np.sqrt(x ** 2 + y ** 2)[:h, :w, np.newaxis]
        fog_pattern = 0.8 + 0.2 * dist
        out = img_np * (1.0 - c) + c * fog_pattern
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_brightness(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Simulates high glare and intense sunlight exposure."""
        severity = max(1, min(5, int(severity)))
        c = [0.10, 0.20, 0.30, 0.40, 0.55][severity - 1]
        return np.clip(img_np + c, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_contrast(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Simulates low contrast caused by overcast lighting or dusk."""
        severity = max(1, min(5, int(severity)))
        c = [0.80, 0.60, 0.40, 0.25, 0.12][severity - 1]
        mean = np.mean(img_np, axis=(0, 1), keepdims=True)
        return np.clip((img_np - mean) * c + mean, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_pixelate(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Simulates digital sensor subsampling and resolution downsampling."""
        severity = max(1, min(5, int(severity)))
        f = [0.80, 0.60, 0.40, 0.25, 0.12][severity - 1]
        h, w = img_np.shape[:2]
        nh = max(1, int(h * f))
        nw = max(1, int(w * f))
        small = cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_LINEAR)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.clip(restored, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def apply_jpeg_compression(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Simulates lossy DCT compression artifacts."""
        severity = max(1, min(5, int(severity)))
        quality = [60, 40, 25, 15, 5][severity - 1]
        uint8_img = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(uint8_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        decompressed = Image.open(buffer).convert("RGB")
        return (np.array(decompressed, dtype=np.float32) / 255.0).astype(np.float32)

    @staticmethod
    def apply_elastic_transform(img_np: np.ndarray, severity: int = 1) -> np.ndarray:
        """Simulates optical lens distortion via elastic field deformation."""
        severity = max(1, min(5, int(severity)))
        alpha, sigma = [
            (4.0, 1.0),
            (8.0, 1.5),
            (12.0, 2.0),
            (16.0, 2.5),
            (20.0, 3.0)
        ][severity - 1]

        h, w = img_np.shape[:2]
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2.0 - 1.0).astype(np.float32),
            (7, 7),
            sigma
        ) * alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2.0 - 1.0).astype(np.float32),
            (7, 7),
            sigma
        ) * alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        distorted = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return np.clip(distorted, 0.0, 1.0).astype(np.float32)

    @classmethod
    def apply_noise(cls, img_np: np.ndarray, noise_type: str, severity: int = 1) -> np.ndarray:
        """
        Dispatches corruption operator by noise name.
        Normalizes aliases to canonical names.
        """
        if noise_type is None:
            return img_np

        canonical_lookup = {
            "gaussian": cls.apply_gaussian_noise,
            "gaussian noise": cls.apply_gaussian_noise,
            "shot": cls.apply_shot_noise,
            "shot noise": cls.apply_shot_noise,
            "impulse": cls.apply_impulse_noise,
            "impulse noise": cls.apply_impulse_noise,
            "defocus blur": cls.apply_defocus_blur,
            "defocus": cls.apply_defocus_blur,
            "motion blur": cls.apply_motion_blur,
            "motion": cls.apply_motion_blur,
            "zoom blur": cls.apply_zoom_blur,
            "zoom": cls.apply_zoom_blur,
            "snow": cls.apply_snow,
            "frost": cls.apply_frost,
            "fog": cls.apply_fog,
            "brightness": cls.apply_brightness,
            "contrast": cls.apply_contrast,
            "pixelate": cls.apply_pixelate,
            "jpeg compression": cls.apply_jpeg_compression,
            "jpeg": cls.apply_jpeg_compression,
            "elastic transform": cls.apply_elastic_transform,
            "elastic": cls.apply_elastic_transform,
        }

        key = str(noise_type).strip().lower()
        if key not in canonical_lookup:
            raise ValueError(f"Unknown corruption type: '{noise_type}'. Supported: {list(canonical_lookup.keys())}")

        return canonical_lookup[key](img_np, severity=severity)

