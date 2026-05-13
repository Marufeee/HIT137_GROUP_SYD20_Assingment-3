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

"""
GUI composition and display helper methods.

- This draws the window: buttons up top, two big image areas side by side.
- It does NOT know how to open a file dialog or handle clicks end-to-end — 
- Think of `SpotTheDifferenceView` as a skeleton: the bones exist, but `load_image`,
  `reveal_unfound`, and `_on_modified_click` are still incomplete

Dependency note:
- We import `GameStats` and `ImageDifferenceEngine` . Those objects hold
  all the actual game data; this class just displays them and forwards user actions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from project_part1_65 import GameStats, ImageDifferenceEngine


class SpotTheDifferenceView:
    """
    Base window class — builds widgets and knows how to resize/draw images.

    Subclass adds: picking a file, reacting to clicks, win/lose popups.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Spot the Difference - HIT137")
        self.root.geometry("1320x760")
        self.root.minsize(1040, 640)

        # These two are the "model" — engine = pictures + hidden boxes, stats = score.
        self.engine = ImageDifferenceEngine()
        self.stats = GameStats()

        # After we resize images to fit the canvas, we need to map click coords back to
        # full-resolution image coords — _scale_x/_scale_y do that (set in _prepare_display_pair).
        self._display_original: Optional[np.ndarray] = None
        self._display_modified: Optional[np.ndarray] = None
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0
        self._can_guess = False

        # Tkinter quirk: you must keep a reference to PhotoImage or it gets garbage-collected
        # and you get a blank canvas. Hence these instance attributes.
        self._photo_left = None
        self._photo_right = None

        self._build_ui()
        self._refresh_labels()

    def _build_ui(self) -> None:
        """Pack all the controls and canvases — order matters for left-to-right layout."""
        control_frame = tk.Frame(self.root, padx=8, pady=8)
        control_frame.pack(fill="x")

        load_btn = tk.Button(control_frame, text="Load Image", command=self.load_image, width=14)
        load_btn.pack(side="left", padx=4)

        reveal_btn = tk.Button(control_frame, text="Reveal Unfound", command=self.reveal_unfound, width=14)
        reveal_btn.pack(side="left", padx=4)

        # StringVar ties the dropdown to actual Python string state, .get() on load.
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

        # Below the toolbar: two columns that grow with the window (expand=True).
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

        # Crosshair cursor hints "this is the interactive side".
        self.canvas_right = tk.Canvas(right_panel, bg="#e8e8e8", cursor="crosshair")
        self.canvas_right.pack(fill="both", expand=True, padx=(6, 0))
        self.canvas_right.bind("<Button-1>", self._on_modified_click)

    def _refresh_labels(self) -> None:
        """Sync text labels with whatever the engine/stats currently say — call after any game event."""
        remaining = 0
        if self.engine.regions:
            remaining = sum(1 for r in self.engine.regions if not r.found)

        self.label_remaining.config(text=f"Remaining: {remaining}")
        self.label_mistakes.config(text=f"Mistakes: {self.stats.mistakes} / {self.stats.max_mistakes}")
        self.label_total.config(text=f"Total Found: {self.stats.total_found}")

    def _prepare_display_pair(
        self, left_img: np.ndarray, right_img: np.ndarray, max_w: int = 620, max_h: int = 620
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shrink both images by the same factor so they fit the canvas without stretching weirdly.

        We also stash _scale_x/_scale_y so clicks on the canvas can be multiplied back to
        "real" image coordinates. The 12-pixel offset in part 3 matches create_image(12,12,...).
        """
        h, w = left_img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        self._scale_x = w / new_w
        self._scale_y = h / new_h

        resized_left = cv2.resize(left_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_right = cv2.resize(right_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_left, resized_right

    def _draw_canvas_images(self) -> None:
        """
        Push numpy BGR arrays onto the canvases as Tk PhotoImages.

        OpenCV lives in BGR land; Tkinter/PIL want RGB — hence cvtColor.
        """
        if self._display_original is None or self._display_modified is None:
            return

        left_rgb = cv2.cvtColor(self._display_original, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(self._display_modified, cv2.COLOR_BGR2RGB)

        self._photo_left = ImageTk.PhotoImage(Image.fromarray(left_rgb))
        self._photo_right = ImageTk.PhotoImage(Image.fromarray(right_rgb))

        self.canvas_left.delete("all")
        self.canvas_right.delete("all")
        self.canvas_left.create_image(12, 12, anchor="nw", image=self._photo_left)
        self.canvas_right.create_image(12, 12, anchor="nw", image=self._photo_right)

        self.canvas_left.config(scrollregion=self.canvas_left.bbox("all"))
        self.canvas_right.config(scrollregion=self.canvas_right.bbox("all"))

    # -------------------------------------------------------------------------
    # Stubs — subclass replaces these with real implementations.
    # If you accidentally run THIS file as main, you'd hit these — run part 3 instead.
    # -------------------------------------------------------------------------

    def load_image(self) -> None:
        raise NotImplementedError

    def reveal_unfound(self) -> None:
        raise NotImplementedError

    def _on_modified_click(self, event: tk.Event) -> None:
        raise NotImplementedError


"""
Interaction logic, round flow, and executable app entry.


- We subclass `SpotTheDifferenceView` and finally implement the three methods that were
  stubbed out: load a file, handle clicks, reveal answers, show popups when you win or
  run out of mistakes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

from project_part2_17_5 import SpotTheDifferenceView


class SpotTheDifferenceApp(SpotTheDifferenceView):
    """
    The playable app — same UI as the parent, but now buttons actually do things.
    """

    def load_image(self) -> None:
        """
        Open a file picker, then tell the engine to load + bake in 5 differences.

        Order matters: set_mode first so load_and_generate uses Normal vs Easy 
        If anything doesnt work (bad path, tiny image, couldn't place 5 boxes), we show
        an error dialog and exit without half-updating the UI.
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
        Copy fresh pixels from the engine, draw circles on TOP for found/revealed spots,
        then resize for display.

        Why copy? We don't want to permanently draw circles on the engine's internal
        arrays — those are the "source of truth" for the round. We only decorate for display.
        """
        original = self.engine.original.copy()
        modified = self.engine.modified.copy()

        for region in self.engine.regions:
            cx, cy = region.center()
            radius = int(max(region.w, region.h) * 0.55)
            if region.found:
                # BGR: red = (0,0,255) in OpenCV's backwards channel order
                cv2.circle(original, (cx, cy), radius, (0, 0, 255), 2)
                cv2.circle(modified, (cx, cy), radius, (0, 0, 255), 2)
            elif region.revealed:
                # Blue-ish for "you cheated / reveal button" — still BGR so (255,0,0) is blue
                cv2.circle(original, (cx, cy), radius, (255, 0, 0), 2)
                cv2.circle(modified, (cx, cy), radius, (255, 0, 0), 2)

        self._display_original, self._display_modified = self._prepare_display_pair(original, modified)
        self._draw_canvas_images()

    def _on_modified_click(self, event: tk.Event) -> None:
        """
        Translate canvas click → image pixel, see if it hit a hidden region.

        The `- 12` matches where we painted the image (offset from canvas edge). If you
        ever change that offset in part 2's _draw_canvas_images, update it here too or
        clicks will feel "off".
        """
        if not self.engine.regions or not self._can_guess or not self.stats.guesses_allowed():
            return

        img_x = int((event.x - 12) * self._scale_x)
        img_y = int((event.y - 12) * self._scale_y)
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
        Win: zero remaining unfound → popup + lock guesses.
        Lose the round: 3 misses → popup with how many you got, lock guesses.

        "Lock guesses" = _can_guess False so further clicks are ignored until new image.
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
        Spoiler button: mark every still-hidden region as revealed, redraw with blue rings,
        and stop counting further clicks as valid guesses.
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
    """Standard Tk entry: one root window, one app object, hand control to the event loop."""
    root = tk.Tk()
    app = SpotTheDifferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
