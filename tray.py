"""System tray icon (pystray): pause / resume / quit and a colour that follows the state."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from PIL import Image, ImageDraw

from state_machine import State

logger = logging.getLogger(__name__)

PAUSED_COLOR = (130, 130, 130)
STATE_RGB: dict[State, tuple[int, int, int]] = {
    State.SECURE: (46, 160, 67),
    State.SUSPECT: (240, 200, 0),
    State.WARNING: (240, 140, 0),
    State.ALERT: (220, 40, 40),
    State.LOCKED: (160, 0, 160),
    State.RECOVERY: (60, 140, 230),
}


class Tray(Protocol):
    """What the headless runner needs from a tray icon."""

    def start(self, on_toggle_pause: Callable[[], None], on_quit: Callable[[], None]) -> None:
        """Show the icon and wire the menu callbacks."""
        ...

    def set_status(self, state: State, paused: bool) -> None:
        """Update icon colour and tooltip."""
        ...

    def stop(self) -> None:
        """Remove the icon."""
        ...


def icon_color(state: State, paused: bool) -> tuple[int, int, int]:
    """Colour of the tray icon for ``state`` (grey while paused)."""
    return PAUSED_COLOR if paused else STATE_RGB[state]


def make_icon(color: tuple[int, int, int]) -> Image.Image:
    """Draw a simple 64x64 round icon with a white eye-like centre."""
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((4, 4, 60, 60), fill=color + (255,))
    draw.ellipse((22, 22, 42, 42), fill=(255, 255, 255, 255))
    draw.ellipse((28, 28, 36, 36), fill=(20, 20, 20, 255))
    return image


class NullTray:
    """Used with ``--no-tray`` or when no tray is available."""

    def start(self, on_toggle_pause: Callable[[], None], on_quit: Callable[[], None]) -> None:
        """Nothing to show."""

    def set_status(self, state: State, paused: bool) -> None:
        """Nothing to update."""

    def stop(self) -> None:
        """Nothing to remove."""


class PystrayTray:
    """Real tray icon; construction fails with ``RuntimeError`` if pystray is not usable."""

    TITLE = "Shoulder Surfing Detector"

    def __init__(self) -> None:
        try:
            import pystray
        except Exception as exc:  # noqa: BLE001 - pystray raises backend-specific errors on import
            raise RuntimeError(f"pystray is unavailable: {exc}") from exc
        self._pystray: Any = pystray
        self._icon: Any = None
        self._paused = False

    def start(self, on_toggle_pause: Callable[[], None], on_quit: Callable[[], None]) -> None:
        """Show the icon in a background thread."""
        pystray = self._pystray

        def toggle(_icon: Any, _item: Any) -> None:
            on_toggle_pause()

        def quit_app(_icon: Any, _item: Any) -> None:
            on_quit()

        menu = pystray.Menu(
            pystray.MenuItem(lambda _item: "Resume" if self._paused else "Pause", toggle, default=True),
            pystray.MenuItem("Quit", quit_app),
        )
        self._icon = pystray.Icon("ssd", make_icon(icon_color(State.SECURE, False)), self.TITLE, menu)
        self._icon.run_detached()

    def set_status(self, state: State, paused: bool) -> None:
        """Recolour the icon and update the tooltip."""
        self._paused = paused
        if self._icon is None:
            return
        self._icon.icon = make_icon(icon_color(state, paused))
        self._icon.title = f"{self.TITLE} - {'PAUSED' if paused else state.value}"
        self._icon.update_menu()

    def stop(self) -> None:
        """Remove the icon."""
        if self._icon is not None:
            self._icon.stop()
            self._icon = None


def create_tray(enabled: bool) -> Tray:
    """Return a working tray, or a :class:`NullTray` (with a log line) when that is impossible."""
    if not enabled:
        return NullTray()
    try:
        return PystrayTray()
    except RuntimeError as exc:
        logger.warning("No tray icon: %s", exc)
        return NullTray()
