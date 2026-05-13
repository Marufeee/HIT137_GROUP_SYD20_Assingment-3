from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox


# DifferenceRegion — one rectangle where we hid a change
# Each "spot the difference" spot is really just a box: top-left (x,y) and size (w,h).
# We also track whether the player found it yet, or whether they hit "reveal" so we
# can draw blue circles instead of red ones.


@dataclass
class DifferenceRegion:
    x: int
    y: int
    w: int
    h: int
    found: bool = False
    revealed: bool = False

    def center(self) -> Tuple[int, int]:
        """Middle of the box — handy when we want to draw a circle around the spot."""
        return (self.x + self.w // 2, self.y + self.h // 2)

    def overlaps(self, other: "DifferenceRegion", padding: int = 15) -> bool:
        """
        Would this box sit on top of another box (or too close)?

        We inflate *this* region by `padding` pixels on each side when checking. That way
        two differences aren't jammed right next to each other — a bit of breathing room
        makes the game fairer and the image less of a mess.
        """
        left_a = self.x - padding
        right_a = self.x + self.w + padding
        top_a = self.y - padding
        bottom_a = self.y + self.h + padding

        left_b = other.x
        right_b = other.x + other.w
        top_b = other.y
        bottom_b = other.y + other.h
        # Classic "two rectangles intersect" logic — if they don't overlap, return False.
        return not (right_a < left_b or right_b < left_a or bottom_a < top_b or bottom_b < top_a)

    def contains_click(self, px: int, py: int, tolerance: int = 18) -> bool:
        """
        Did the player click inside (or near) this region?

        We use tolerance because clicks are chunky — nobody hits the exact pixel. So we
        pretend the box is a little bigger than it really is for hit-testing only.
        """
        return (
            self.x - tolerance <= px <= self.x + self.w + tolerance
            and self.y - tolerance <= py <= self.y + self.h + tolerance
        )


# DifferenceOperation — the OOP "template" for any kind of image tweak
# This is an abstract base class: you never instantiate DifferenceOperation itself.
# You subclass it and implement apply(). The engine then picks random subclasses and
# calls .apply() — that's polymorphism (same method name, different behaviour).


class DifferenceOperation(ABC):
    @abstractmethod
    def apply(self, image: np.ndarray, region: DifferenceRegion) -> None:
        """Actually change the pixels inside `region` on `image` (BGR numpy array)."""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Human-readable name — useful if you ever want to debug or log what ran."""
        raise NotImplementedError


class ColorShiftOperation(DifferenceOperation):
    """
    Normal mode: slap a coloured tint over the patch.

    We blend the original patch with a random solid colour at 45% strength. That tends
    to read as "that chunk looks wrong" without needing to explain hue/saturation math.
    """

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> None:
        patch = image[region.y : region.y + region.h, region.x : region.x + region.w]
        tint = np.array(
            [
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ],
            dtype=np.uint8,
        )
        overlay = np.full_like(patch, tint)
        alpha = 0.45
        image[region.y : region.y + region.h, region.x : region.x + region.w] = cv2.addWeighted(
            patch, 1.0 - alpha, overlay, alpha, 0.0
        )

    def name(self) -> str:
        return "Color Shift"


class BlurPatchOperation(DifferenceOperation):
    """
    Normal mode: Gaussian blur — the area goes soft and dreamy compared to the original.

    OpenCV does the heavy lifting; bigger kernel = more obvious blur.
    """

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> None:
        patch = image[region.y : region.y + region.h, region.x : region.x + region.w]
        image[region.y : region.y + region.h, region.x : region.x + region.w] = cv2.GaussianBlur(
            patch, (21, 21), sigmaX=4.2
        )

    def name(self) -> str:
        return "Blur Patch"


class NoisePatchOperation(DifferenceOperation):
    """
    Normal mode: sprinkle random +/- brightness on each pixel.

    Looks grainy / staticky. Fine detail (grass, water) can hide it a bit, but next to
    the original it's usually visible.
    """

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> None:
        patch = image[region.y : region.y + region.h, region.x : region.x + region.w]
        noise = np.random.randint(-70, 71, patch.shape, dtype=np.int16)
        image[region.y : region.y + region.h, region.x : region.x + region.w] = np.clip(
            patch.astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)

    def name(self) -> str:
        return "Noise Patch"


class InvertPatchOperation(DifferenceOperation):
    """
    Easy mode: photographic negative inside the box (255 - pixel).

    Hard to miss — great for demos or when you want the player to win fast.
    """

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> None:
        patch = image[region.y : region.y + region.h, region.x : region.x + region.w]
        image[region.y : region.y + region.h, region.x : region.x + region.w] = 255 - patch

    def name(self) -> str:
        return "Invert Patch"


class PixelatePatchOperation(DifferenceOperation):
    """
    Easy mode: Minecraft / mosaic look.

    Trick: shrink the patch to tiny, then blow it back up with nearest-neighbour so you
    get chunky blocks instead of smooth pixels.
    """

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> None:
        patch = image[region.y : region.y + region.h, region.x : region.x + region.w]
        h, w = patch.shape[:2]
        small_w = max(2, w // 10)
        small_h = max(2, h // 10)
        down = cv2.resize(patch, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
        image[region.y : region.y + region.h, region.x : region.x + region.w] = pixelated

    def name(self) -> str:
        return "Pixelate Patch"


class BrightnessTintOperation(DifferenceOperation):
    """
    Easy mode: push hue / saturation / value in HSV space then convert back to BGR.

    Reads as "that patch is the wrong colour AND too bright" — very obvious next to
    the untouched original.
    """

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> None:
        patch = image[region.y : region.y + region.h, region.x : region.x + region.w]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + 40) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + 80, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + 95, 0, 255)
        tinted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        image[region.y : region.y + region.h, region.x : region.x + region.w] = tinted

    def name(self) -> str:
        return "Brightness Tint"



# ImageDifferenceEngine — load image, clone it, carve out 5 regions, apply ops
# Flow for someone reading top-to-bottom:
# 1. Optional seed (if you ever want reproducible randomness for testing).
# 2. set_mode("Normal" or "Easy") swaps which three operation classes we use.
# 3. load_and_generate(path) reads disk, copies pixels, builds regions, runs apply().


class ImageDifferenceEngine:
    def __init__(self, rng_seed: Optional[int] = None) -> None:
        self._rng_seed = rng_seed
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        self._mode: str = "Normal"
        self._operations: Sequence[DifferenceOperation] = self._operations_for_mode(self._mode)
        self._original_image: Optional[np.ndarray] = None
        self._modified_image: Optional[np.ndarray] = None
        self._regions: List[DifferenceRegion] = []

    @property
    def mode(self) -> str:
        """Which flavour we're in — GUI shows this after load."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """
        Called from the GUI before loading. Accepts "Easy" / "easy" / "easy mode" etc.

        We normalise to "Easy" or "Normal" internally so the rest of the code stays simple.
        """
        normalized = (mode or "").strip().lower()
        if normalized in ("easy", "easy mode"):
            self._mode = "Easy"
        else:
            self._mode = "Normal"
        self._operations = self._operations_for_mode(self._mode)

    def _operations_for_mode(self, mode: str) -> Sequence[DifferenceOperation]:
        """Pick the trio of operation objects for this mode (new instances each time)."""
        if mode == "Easy":
            return (
                InvertPatchOperation(),
                PixelatePatchOperation(),
                BrightnessTintOperation(),
            )
        return (
            ColorShiftOperation(),
            BlurPatchOperation(),
            NoisePatchOperation(),
        )

    @property
    def original(self) -> np.ndarray:
        if self._original_image is None:
            raise ValueError("No image loaded.")
        return self._original_image

    @property
    def modified(self) -> np.ndarray:
        if self._modified_image is None:
            raise ValueError("No image loaded.")
        return self._modified_image

    @property
    def regions(self) -> List[DifferenceRegion]:
        """The five (or however many) hidden boxes — GUI uses this for clicks and drawing."""
        return self._regions

    def load_and_generate(self, image_path: str, total_differences: int = 5) -> None:
        """
        The main event: read file, validate size, clone, scatter differences.

        We shuffle the operation list then cycle through it so you don't randomly get
        five blurs in a row (which would be boring / hard to tell apart).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image. Please choose JPG/PNG/BMP.")

        h, w = image.shape[:2]
        if min(h, w) < 120:
            raise ValueError("Image is too small. Please choose a larger image.")

        self._original_image = image
        self._modified_image = image.copy()
        self._regions = self._create_non_overlapping_regions(w, h, total_differences)

        ops = list(self._operations)
        random.shuffle(ops)
        for idx, region in enumerate(self._regions):
            ops[idx % len(ops)].apply(self._modified_image, region)

    def _create_non_overlapping_regions(self, width: int, height: int, count: int) -> List[DifferenceRegion]:
        """
        Throw darts at random rectangles until we have `count` that don't overlap.

        If the image is weird or tiny we might fail after max_attempts — then we bail
        with a clear error so the GUI can show a dialog instead of hanging forever.
        """
        created: List[DifferenceRegion] = []
        attempts = 0
        max_attempts = 4000

        min_size = max(36, min(width, height) // 9)
        max_size = max(min_size + 12, min(width, height) // 4)

        while len(created) < count and attempts < max_attempts:
            attempts += 1
            rw = random.randint(min_size, max_size)
            rh = random.randint(min_size, max_size)
            if rw >= width - 5 or rh >= height - 5:
                continue

            rx = random.randint(4, width - rw - 4)
            ry = random.randint(4, height - rh - 4)
            candidate = DifferenceRegion(rx, ry, rw, rh)
            if all(not candidate.overlaps(existing) for existing in created):
                created.append(candidate)

        if len(created) != count:
            raise ValueError("Could not generate non-overlapping differences. Try another image.")
        return created


# GameStats — boring but important: how many wrong clicks, how many found overall
# Kept separate from the engine on purpose: image stuff vs score stuff = cleaner design.


class GameStats:
    def __init__(self) -> None:
        self._total_found = 0
        self._images_played = 0
        self._mistakes_current_image = 0
        self._max_mistakes = 3

    @property
    def total_found(self) -> int:
        """Lifetime counter across multiple loaded images (assignment asked for cumulative)."""
        return self._total_found

    @property
    def images_played(self) -> int:
        return self._images_played

    @property
    def mistakes(self) -> int:
        """Wrong clicks *this* round only — resets when you load a new picture."""
        return self._mistakes_current_image

    @property
    def max_mistakes(self) -> int:
        return self._max_mistakes

    def add_found(self, amount: int = 1) -> None:
        self._total_found += amount

    def add_mistake(self) -> None:
        self._mistakes_current_image += 1

    def reset_for_new_image(self) -> None:
        """New image = new mistake budget, but we bump images_played for stats nerds."""
        self._images_played += 1
        self._mistakes_current_image = 0

    def guesses_allowed(self) -> bool:
        """False once you've burned through 3 misses — GUI should ignore further clicks."""
        return self._mistakes_current_image < self._max_mistakes

