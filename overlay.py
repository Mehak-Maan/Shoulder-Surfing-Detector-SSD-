"""Fullscreen blur overlay: screenshot (mss) -> Gaussian blur (OpenCV) -> borderless topmost window (Tk).

The screenshot is taken *before* the window appears, so the overlay never captures itself.
Tk runs in its own thread; on macOS Tk insists on the main thread, so the overlay is
best-effort there (see README, limitations).
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from config import OverlayConfig
from detector import Image, as_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Screenshot:
    """A BGR capture of the whole virtual desktop and the position of its top-left corner."""

    image: Image
    left: int
    top: int


class ScreenGrabber(Protocol):
    """Captures the screen."""

    def grab(self) -> Screenshot:
        """Return a screenshot of all monitors."""
        ...


class OverlayWindow(Protocol):
    """A borderless topmost window that shows one image."""

    def show(self, image: Image, left: int, top: int) -> None:
        """Display ``image`` with its top-left corner at ``(left, top)``."""
        ...

    def hide(self) -> None:
        """Remove the window."""
        ...

    def close(self) -> None:
        """Release all resources."""
        ...


class MssGrabber:
    """Screen capture through the ``mss`` library."""

    def __init__(self) -> None:
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError("The blur overlay needs the 'mss' package (pip install mss).") from exc
        self._mss: Any = mss

    def grab(self) -> Screenshot:
        """Capture the virtual screen that spans every monitor."""
        with self._mss.mss() as sct:
            monitor = sct.monitors[0]
            raw = sct.grab(monitor)
            bgra = np.asarray(raw, dtype=np.uint8)
            bgr = as_image(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR))
            return Screenshot(bgr, int(monitor["left"]), int(monitor["top"]))


def make_blurred(image: Image, cfg: OverlayConfig) -> Image:
    """Return a heavily blurred, darkened copy of ``image`` with the overlay message on it."""
    height, width = image.shape[:2]
    small = cv2.resize(
        image, (max(1, width // cfg.blur_downscale), max(1, height // cfg.blur_downscale)), interpolation=cv2.INTER_AREA
    )
    blurred = cv2.GaussianBlur(small, (cfg.blur_kernel, cfg.blur_kernel), 0)
    out = as_image(cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR))
    if cfg.darken > 0:
        out = as_image(cv2.convertScaleAbs(out, alpha=1.0 - cfg.darken))
    scale = max(0.8, width / 1600)
    (tw, _), _ = cv2.getTextSize(cfg.message, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    origin = ((width - tw) // 2, height // 2)
    cv2.putText(out, cfg.message, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class TkOverlayWindow:
    """Borderless, topmost, fullscreen Tk window living in its own thread."""

    def __init__(self) -> None:
        self._commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._ready = threading.Event()
        self._failed: str | None = None
        self._thread = threading.Thread(target=self._run, name="overlay-tk", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0) or self._failed:
            raise RuntimeError(f"Overlay window could not start: {self._failed or 'timeout'}")

    def show(self, image: Image, left: int, top: int) -> None:
        """Queue showing ``image`` at ``(left, top)``."""
        ok, png = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("PNG encoding of the overlay image failed")
        self._commands.put(("show", (base64.b64encode(png.tobytes()), image.shape[1], image.shape[0], left, top)))

    def hide(self) -> None:
        """Queue removing the window."""
        self._commands.put(("hide", None))

    def close(self) -> None:
        """Stop the Tk thread."""
        self._commands.put(("quit", None))
        self._thread.join(timeout=3.0)

    def _run(self) -> None:  # pragma: no cover - needs a display
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
        except Exception as exc:  # noqa: BLE001 - any GUI start-up failure must be reported, not raised in a thread
            self._failed = str(exc)
            self._ready.set()
            return
        state: dict[str, Any] = {"win": None, "photo": None}

        def destroy() -> None:
            if state["win"] is not None:
                state["win"].destroy()
                state["win"] = None
                state["photo"] = None

        def poll() -> None:
            try:
                while True:
                    name, payload = self._commands.get_nowait()
                    if name == "show":
                        data, width, height, left, top = payload
                        destroy()
                        win = tk.Toplevel(root)
                        win.overrideredirect(True)
                        win.geometry(f"{width}x{height}+{left}+{top}")
                        win.attributes("-topmost", True)
                        photo = tk.PhotoImage(data=data)
                        tk.Label(win, image=photo, borderwidth=0).pack()
                        win.update_idletasks()
                        state["win"], state["photo"] = win, photo
                    elif name == "hide":
                        destroy()
                    elif name == "quit":
                        destroy()
                        root.destroy()
                        return
            except queue.Empty:
                pass
            root.after(30, poll)

        root.after(30, poll)
        self._ready.set()
        root.mainloop()


class NullOverlay:
    """Overlay used in dry-run mode: records what would have happened."""

    def __init__(self) -> None:
        self.visible = False

    def show(self) -> None:
        """Pretend to show the overlay."""
        self.visible = True
        logger.info("[dry-run] blur overlay would be shown")

    def hide(self) -> None:
        """Pretend to hide the overlay."""
        if self.visible:
            logger.info("[dry-run] blur overlay would be hidden")
        self.visible = False

    def close(self) -> None:
        """Nothing to release."""


class BlurOverlay:
    """Shows a blurred snapshot of the desktop above everything else."""

    def __init__(self, cfg: OverlayConfig, grabber: ScreenGrabber, window: OverlayWindow) -> None:
        self._cfg = cfg
        self._grabber = grabber
        self._window = window
        self.visible = False

    def show(self) -> None:
        """Capture, blur and display the desktop (no-op if already visible)."""
        if self.visible:
            return
        shot = self._grabber.grab()  # before the window exists, so it cannot capture itself
        self._window.show(make_blurred(shot.image, self._cfg), shot.left, shot.top)
        self.visible = True
        logger.info("Blur overlay shown")

    def hide(self) -> None:
        """Remove the overlay (no-op if hidden)."""
        if not self.visible:
            return
        self._window.hide()
        self.visible = False
        logger.info("Blur overlay hidden")

    def close(self) -> None:
        """Hide the overlay and release the window."""
        self.hide()
        self._window.close()
