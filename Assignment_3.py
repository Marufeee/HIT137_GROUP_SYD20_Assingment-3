"""
Spot the Difference Game - HIT137 Assignment 3

How the code is organised (read top → bottom):
    1) Constants and type aliases — easy-to-change settings used across the program
    2) DifferenceRegion          — one hidden “spot” as a rectangle + found/reveal flags
    3) DifferenceOperation + subclasses — OpenCV tricks that change a patch (OOP / polymorphism)
    4) ImageDifferenceEngine     — load image, clone it, place 5 non-overlapping regions, apply ops
    5) GameStats                 — mistakes + total found (kept separate from pixels on purpose)
    6) SpotTheDifferenceView     — Tk window, labels, canvases, scaling + fullscreen resize logic
    7) SpotTheDifferenceApp      — subclass: file picker, clicks, win/lose, reveal button
    8) main()                    — boots Tk and starts the event loop

Program Flow:
    1) User loads an image.
    2) The engine creates random non-overlapping regions.
    3) DifferenceOperation subclasses modify only those selected regions.
    4) Original and modified images are shown side-by-side.
    5) User clicks the modified image to find differences.
    6) GameStats updates remaining differences, total found, and mistakes.
    7) The reveal button highlights unfound regions and ends the current round.

OOP Concepts Used:
    - Encapsulation: each class manages its own data and behaviour.
    - Abstraction: DifferenceOperation defines a common interface for all effects.
    - Inheritance: each operation class inherits from DifferenceOperation.
    - Polymorphism: the engine calls apply() without knowing which effect class is used.
    - Separation of Concerns: image processing, game stats, and UI logic are separated.

Class Relationships:
    SpotTheDifferenceApp  → inherits from → SpotTheDifferenceView
    ImageDifferenceEngine → uses         → DifferenceOperation subclasses
    GameStats             → stores       → gameplay statistics
    DifferenceRegion      → stores       → geometry and click-hit testing data

Features:
    - Random non-overlapping differences
    - Two difficulty modes: Normal and Easy
    - Dynamic image scaling while keeping aspect ratio
    - Click detection with tolerance
    - Red circles for found differences
    - Blue circles for revealed differences
    - Mistake tracking and lockout after maximum mistakes
    - Resize/fullscreen support
                  — boots Tk and starts the event loop
"""

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


# =============================================================================
# SECTION 1 — DifferenceRegion (geometry + hit testing for one hidden spot)
# =============================================================================
# The game is really “five rectangles on the modified image”. Each DifferenceRegion stores
# where that rectangle is and whether the player already found it or hit Reveal.


@dataclass
class DifferenceRegion:
    x: int
    y: int
    w: int
    h: int
    found: bool = False
    revealed: bool = False

    def center(self) -> Tuple[int, int]:
        """Pixel coords of the middle — we draw circles around here when you find/reveal."""
        return (self.x + self.w // 2, self.y + self.h // 2)

    def overlaps(self, other: "DifferenceRegion", padding: int = 15) -> bool:
        """
        True if this box overlaps `other` (counting a few extra pixels of padding).

        Padding stops two differences from spawning right on top of each other, which
        would be unfair and look messy.
        """
        left_a = self.x - padding
        right_a = self.x + self.w + padding
        top_a = self.y - padding
        bottom_a = self.y + self.h + padding

        left_b = other.x
        right_b = other.x + other.w
        top_b = other.y
        bottom_b = other.y + other.h
        return not (right_a < left_b or right_b < left_a or bottom_a < top_b or bottom_b < top_a)

    def contains_click(self, px: int, py: int, tolerance: int = 18) -> bool:
        """
        True if (px, py) falls inside the region, with a bit of slack.

        Mouse clicks aren’t pixel-perfect; tolerance grows the hit box for gameplay feel.
        """
        return (
            self.x - tolerance <= px <= self.x + self.w + tolerance
            and self.y - tolerance <= py <= self.y + self.h + tolerance
        )


# =============================================================================
# SECTION 2 — DifferenceOperation (abstract “recipe” for one kind of visual tweak)
# =============================================================================
# Inheritance + polymorphism: the engine holds a list of operation objects and calls
# .apply() without caring which concrete class it is.


class DifferenceOperation(ABC):
    """Subclass this for each effect type; never instantiate this base class directly."""

    @abstractmethod
    def apply(self, image: np.ndarray, region: DifferenceRegion) -> None:
        """Mutate `image` inside the rectangle `region` (BGR uint8 array)."""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Short label — handy for debugging or future logging."""
        raise NotImplementedError


class ColorShiftOperation(DifferenceOperation):
    """
    Normal mode: random colour tint blended over the patch.

    45% blend with a random BGR tint — usually obvious when you compare left vs right.
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
    """Normal mode: Gaussian blur — patch goes soft while the rest stays sharp."""

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> None:
        patch = image[region.y : region.y + region.h, region.x : region.x + region.w]
        image[region.y : region.y + region.h, region.x : region.x + region.w] = cv2.GaussianBlur(
            patch, (21, 21), sigmaX=4.2
        )

    def name(self) -> str:
        return "Blur Patch"


class NoisePatchOperation(DifferenceOperation):
    """Normal mode: add random per-pixel noise — looks grainy / static-y."""

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> None:
        patch = image[region.y : region.y + region.h, region.x : region.x + region.w]
        noise = np.random.randint(-70, 71, patch.shape, dtype=np.int16)
        image[region.y : region.y + region.h, region.x : region.x + region.w] = np.clip(
            patch.astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)

    def name(self) -> str:
        return "Noise Patch"


class InvertPatchOperation(DifferenceOperation):
    """Easy mode: invert colours (255 - pixel) — very hard to miss."""

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> None:
        patch = image[region.y : region.y + region.h, region.x : region.x + region.w]
        image[region.y : region.y + region.h, region.x : region.x + region.w] = 255 - patch

    def name(self) -> str:
        return "Invert Patch"


class PixelatePatchOperation(DifferenceOperation):
    """
    Easy mode: chunky “minecraft” blocks.

    Shrink patch → resize back up with nearest-neighbour so pixels snap to a grid.
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
    """Easy mode: big hue/sat/value push in HSV then convert back to BGR — loud colour shift."""

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


# =============================================================================
# SECTION 3 — ImageDifferenceEngine (file I/O + random placement + calling operations)
# =============================================================================


class ImageDifferenceEngine:
    """
    Owns the original BGR image, the modified clone, and the list of DifferenceRegions.

    Normal vs Easy mode swaps which three operation classes get instantiated — see set_mode.
    """

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
        """\"Normal\" or \"Easy\" — the GUI prints this after a successful load."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """
        Call this before load_and_generate. Accepts \"easy\", \"Easy\", \"easy mode\", etc.

        Anything that doesn’t look like easy → treated as Normal.
        """
        normalized = (mode or "").strip().lower()
        if normalized in ("easy", "easy mode"):
            self._mode = "Easy"
        else:
            self._mode = "Normal"
        self._operations = self._operations_for_mode(self._mode)

    def _operations_for_mode(self, mode: str) -> Sequence[DifferenceOperation]:
        """Fresh instances each time — keeps behaviour simple and stateless per op."""
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
        """Empty until load_and_generate succeeds; then length == total_differences (5)."""
        return self._regions

    def load_and_generate(self, image_path: str, total_differences: int = 5) -> None:
        """
        Load from disk, sanity-check size, clone to modified, spawn regions, apply effects.

        We shuffle the op list then cycle so you don’t accidentally get five identical effects.
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
        Randomly propose rectangles until `count` non-overlapping ones exist.

        Gives up after max_attempts with a ValueError so the GUI can show a friendly dialog
        instead of spinning forever on a pathological image.
        """
        created: List[DifferenceRegion] = []
        attempts = 0
        max_attempts = 4000

        # Bigger min/max than tiny thumbnails — differences easier to see and click.
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


# =============================================================================
# SECTION 4 — GameStats (scoreboard stuff, no OpenCV here)
# =============================================================================


class GameStats:
    """Tracks lifetime \"total found\" plus per-image mistakes (max 3)."""

    def __init__(self) -> None:
        self._total_found = 0
        self._images_played = 0
        self._mistakes_current_image = 0
        self._max_mistakes = 3

    @property
    def total_found(self) -> int:
        """Never reset by this class — grows across multiple images you load."""
        return self._total_found

    @property
    def images_played(self) -> int:
        """Incremented each time you start a new image (even if you quit early)."""
        return self._images_played

    @property
    def mistakes(self) -> int:
        """Wrong clicks on the *current* image only."""
        return self._mistakes_current_image

    @property
    def max_mistakes(self) -> int:
        return self._max_mistakes

    def add_found(self, amount: int = 1) -> None:
        self._total_found += amount

    def add_mistake(self) -> None:
        self._mistakes_current_image += 1

    def reset_for_new_image(self) -> None:
        """Call when user loads a new picture: fresh mistake budget, bump images_played."""
        self._images_played += 1
        self._mistakes_current_image = 0

    def guesses_allowed(self) -> bool:
        """Once mistakes hit 3, the GUI should ignore further click attempts."""
        return self._mistakes_current_image < self._max_mistakes


# =============================================================================
# SECTION 5 — SpotTheDifferenceView (Tk layout + image scaling + resize hooks)
# =============================================================================
# This class used to be “Part 2” in a split project. The subclass below overrides the three
# stub methods at the bottom with real behaviour.


class SpotTheDifferenceView:
    """
    Builds the toolbar + two canvases, keeps display buffers in sync with window size.

    Subclass responsibility: load_image, reveal_unfound, _on_modified_click (implemented in App).
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Spot the Difference - HIT137")
        self.root.geometry("1320x760")
        self.root.minsize(1040, 640)

        # “Model” objects — no Tk widgets inside them, just data + OpenCV arrays.
        self.engine = ImageDifferenceEngine()
        self.stats = GameStats()

        # Smaller numpy copies actually painted on screen; _scale_* maps canvas coords → full image.
        self._display_original: Optional[np.ndarray] = None
        self._display_modified: Optional[np.ndarray] = None
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0
        self._can_guess = False

        # Must hold PhotoImage references or Tk silently drops them → blank canvases.
        self._photo_left = None
        self._photo_right = None

        # Bitmap top-left on the *right* canvas (where clicks matter); updated in _draw_canvas_images.
        self._img_off_x = 0
        self._img_off_y = 0

        # after() job id for debouncing resize redraws (fullscreen fires tons of Configure events).
        self._resize_job: Optional[str] = None

        self._build_ui()
        self._refresh_labels()

    def _build_ui(self) -> None:
        """Lay out widgets: top bar first, then two equal columns for original vs modified."""
        control_frame = tk.Frame(self.root, padx=8, pady=8)
        control_frame.pack(fill="x")

        tk.Button(control_frame, text="Load Image", command=self.load_image, width=14).pack(side="left", padx=4)
        tk.Button(control_frame, text="Reveal Unfound", command=self.reveal_unfound, width=14).pack(
            side="left", padx=4
        )

        # Dropdown writes into mode_var; load_image reads it before generating differences.
        self.mode_var = tk.StringVar(value="Normal")
        tk.Label(control_frame, text="Mode:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(14, 4))
        mode_menu = tk.OptionMenu(control_frame, self.mode_var, "Normal", "Easy")
        mode_menu.config(width=8)
        mode_menu.pack(side="left", padx=(0, 14))

        self.label_remaining = tk.Label(control_frame, text="Remaining: -", font=("Segoe UI", 11, "bold"))
        self.label_remaining.pack(side="left", padx=14)

        self.label_mistakes = tk.Label(control_frame, text="Mistakes: 0 / 3", font=("Segoe UI", 11))
        self.label_mistakes.pack(side="left", padx=14)

        self.label_total = tk.Label(control_frame, text="Total Found: 0", font=("Segoe UI", 11))
        self.label_total.pack(side="left", padx=14)

        self.label_message = tk.Label(control_frame, text="Load an image to begin.", fg="navy")
        self.label_message.pack(side="left", padx=14)

        # expand=True so maximizing the window grows these frames (and thus the canvases).
        images_frame = tk.Frame(self.root, padx=8, pady=8)
        images_frame.pack(fill="both", expand=True)

        left_panel = tk.Frame(images_frame)
        left_panel.pack(side="left", fill="both", expand=True)
        right_panel = tk.Frame(images_frame)
        right_panel.pack(side="left", fill="both", expand=True)

        tk.Label(left_panel, text="Original (Reference)", font=("Segoe UI", 10, "bold")).pack(pady=(0, 6))
        tk.Label(right_panel, text="Modified (Click Here)", font=("Segoe UI", 10, "bold")).pack(pady=(0, 6))

        self.canvas_left = tk.Canvas(left_panel, bg="#e8e8e8", cursor="arrow")
        self.canvas_left.pack(fill="both", expand=True, padx=(0, 6))

        self.canvas_right = tk.Canvas(right_panel, bg="#e8e8e8", cursor="crosshair")
        self.canvas_right.pack(fill="both", expand=True, padx=(6, 0))
        self.canvas_right.bind("<Button-1>", self._on_modified_click)

        # Any resize on either pane triggers a debounced redraw (only matters once an image is loaded).
        self.canvas_left.bind("<Configure>", self._on_canvas_configure)
        self.canvas_right.bind("<Configure>", self._on_canvas_configure)

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        """Fires when canvas size changes — schedule a single redraw after things settle."""
        if not self.engine.regions:
            return
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self._finish_canvas_resize)

    def _finish_canvas_resize(self) -> None:
        """Runs on the timer: ask subclass to redraw if it implemented _redraw_images."""
        self._resize_job = None
        if not self.engine.regions:
            return
        redraw = getattr(self, "_redraw_images", None)
        if callable(redraw):
            redraw()

    def _get_canvas_display_limits(self) -> Tuple[int, int]:
        """
        Return (max_width, max_height) for the scaled photo inside one canvas.

        Uses live winfo_* so fullscreen works. Tiny values early in startup → safe fallback.
        """
        self.root.update_idletasks()
        margin = 24
        cw = self.canvas_right.winfo_width()
        ch = self.canvas_right.winfo_height()
        if cw < 80 or ch < 80:
            return (620, 620)
        return (max(1, cw - margin), max(1, ch - margin))

    def _refresh_labels(self) -> None:
        """Call after anything changes found/mistake state so the top bar stays truthful."""
        remaining = 0
        if self.engine.regions:
            remaining = sum(1 for r in self.engine.regions if not r.found)

        self.label_remaining.config(text=f"Remaining: {remaining}")
        self.label_mistakes.config(text=f"Mistakes: {self.stats.mistakes} / {self.stats.max_mistakes}")
        self.label_total.config(text=f"Total Found: {self.stats.total_found}")

    def _prepare_display_pair(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        max_w: Optional[int] = None,
        max_h: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize both full-res images identically so they still line up side by side.

        Important: we do NOT cap scale at 1.0 anymore — small photos can grow to fill a
        large monitor. INTER_AREA looks nicer when shrinking; INTER_LINEAR when enlarging.
        """
        if max_w is None or max_h is None:
            max_w, max_h = self._get_canvas_display_limits()

        h, w = left_img.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # How many full-res pixels correspond to one on-screen pixel (for click mapping).
        self._scale_x = w / new_w
        self._scale_y = h / new_h

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized_left = cv2.resize(left_img, (new_w, new_h), interpolation=interp)
        resized_right = cv2.resize(right_img, (new_w, new_h), interpolation=interp)
        return resized_left, resized_right

    def _draw_canvas_images(self) -> None:
        """
        Convert BGR numpy → RGB PhotoImage, center each image inside its canvas, remember offsets.

        _img_off_* must match where we paint on the right canvas or click detection drifts.
        """
        if self._display_original is None or self._display_modified is None:
            return

        left_rgb = cv2.cvtColor(self._display_original, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(self._display_modified, cv2.COLOR_BGR2RGB)

        self._photo_left = ImageTk.PhotoImage(Image.fromarray(left_rgb))
        self._photo_right = ImageTk.PhotoImage(Image.fromarray(right_rgb))

        self.canvas_left.delete("all")
        self.canvas_right.delete("all")

        # shape[:2] is (height, width) in numpy row-major order — don’t mix them up.
        disp_h, disp_w = self._display_original.shape[:2]
        lx = max(0, (self.canvas_left.winfo_width() - disp_w) // 2)
        ly = max(0, (self.canvas_left.winfo_height() - disp_h) // 2)
        rx = max(0, (self.canvas_right.winfo_width() - disp_w) // 2)
        ry = max(0, (self.canvas_right.winfo_height() - disp_h) // 2)
        self._img_off_x, self._img_off_y = rx, ry

        self.canvas_left.create_image(lx, ly, anchor="nw", image=self._photo_left)
        self.canvas_right.create_image(rx, ry, anchor="nw", image=self._photo_right)

        self.canvas_left.config(scrollregion=self.canvas_left.bbox("all"))
        self.canvas_right.config(scrollregion=self.canvas_right.bbox("all"))

    # --- Stubs: subclass (SpotTheDifferenceApp) replaces these with real code ---

    def load_image(self) -> None:
        raise NotImplementedError

    def reveal_unfound(self) -> None:
        raise NotImplementedError

    def _on_modified_click(self, event: tk.Event) -> None:
        raise NotImplementedError


# =============================================================================
# SECTION 6 — SpotTheDifferenceApp (gameplay: subclass of the View above)
# =============================================================================


class SpotTheDifferenceApp(SpotTheDifferenceView):
    """
    Everything that makes the game actually playable lives here.

    Inherits all layout/scaling from SpotTheDifferenceView; overrides the three stubs.
    """

    def load_image(self) -> None:
        """
        File dialog → engine loads + generates 5 differences.

        set_mode runs first so the engine knows Normal vs Easy before touching pixels.
        On failure we show a messagebox and return without touching round state.
        """
        image_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not image_path:
            return

        try:
            self.engine.set_mode(self.mode_var.get())
            self.engine.load_and_generate(image_path, total_differences=5)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        self.stats.reset_for_new_image()
        self._can_guess = True
        self.label_message.config(text=f"Loaded: {Path(image_path).name} ({self.engine.mode} Mode)")
        self._redraw_images()
        self._refresh_labels()

    def _redraw_images(self) -> None:
        """
        Always work on copies — we draw hint circles for display only, not on engine buffers.

        Flow: copy originals → draw circles in full-res coords → downscale for canvas → paint.
        """
        original = self.engine.original.copy()
        modified = self.engine.modified.copy()

        for region in self.engine.regions:
            cx, cy = region.center()
            radius = int(max(region.w, region.h) * 0.55)
            if region.found:
                # OpenCV uses BGR order: “red” is (0, 0, 255).
                cv2.circle(original, (cx, cy), radius, (0, 0, 255), 2)
                cv2.circle(modified, (cx, cy), radius, (0, 0, 255), 2)
            elif region.revealed:
                # BGR blue is (255, 0, 0) — yes, it’s backwards from what you’d guess.
                cv2.circle(original, (cx, cy), radius, (255, 0, 0), 2)
                cv2.circle(modified, (cx, cy), radius, (255, 0, 0), 2)

        self._display_original, self._display_modified = self._prepare_display_pair(original, modified)
        self._draw_canvas_images()

    def _on_modified_click(self, event: tk.Event) -> None:
        """
        Map canvas (event.x, event.y) → full image coords, then ask regions for a hit.

        Subtract _img_off_* first because the bitmap might be centered with empty margin.
        """
        if not self.engine.regions or not self._can_guess or not self.stats.guesses_allowed():
            return

        img_x = int((event.x - self._img_off_x) * self._scale_x)
        img_y = int((event.y - self._img_off_y) * self._scale_y)
        if img_x < 0 or img_y < 0:
            return

        matched = False
        for region in self.engine.regions:
            if not region.found and region.contains_click(img_x, img_y):
                region.found = True
                self.stats.add_found()
                matched = True
                break

        if matched:
            self.label_message.config(text="Great! Difference found.")
        else:
            self.stats.add_mistake()
            self.label_message.config(text="Missed! Try again.")

        self._redraw_images()
        self._refresh_labels()
        self._check_end_conditions()

    def _check_end_conditions(self) -> None:
        """
        Win: zero unfound → congrats + lock input.

        Lose: 3 mistakes → warning + lock input. Either way _can_guess blocks further clicks.
        """
        remaining = sum(1 for r in self.engine.regions if not r.found)

        if remaining == 0:
            self._can_guess = False
            self.label_message.config(text="All 5 differences found. Load another image!")
            messagebox.showinfo("Round Complete", "Excellent! You found all 5 differences.")
            return

        if self.stats.mistakes >= self.stats.max_mistakes:
            self._can_guess = False
            found_count = 5 - remaining
            self.label_message.config(text=f"Too many mistakes (3). Found {found_count}/5. Load new image.")
            messagebox.showwarning(
                "Too Many Mistakes",
                f"You reached 3 mistakes.\nDifferences found: {found_count}/5\nLoad a new image to continue.",
            )

    def reveal_unfound(self) -> None:
        """
        Cheat sheet: mark all still-hidden regions as revealed, draw blue rings, stop guessing.

        Player is expected to load a new image if they want a clean round after this.
        """
        if not self.engine.regions:
            return

        unrevealed = 0
        for region in self.engine.regions:
            if not region.found:
                region.revealed = True
                unrevealed += 1

        self._can_guess = False
        self._redraw_images()
        self._refresh_labels()
        self.label_message.config(text=f"Revealed {unrevealed} unfound differences. Load new image.")


def main() -> None:
    """Entry point: create Tk root, construct app (must keep reference!), start mainloop."""
    root = tk.Tk()
    app = SpotTheDifferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
