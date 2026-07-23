from pathlib import Path
from unittest.mock import Mock, patch

from desktop_launcher import DesktopApi


APP_ROOT = Path(__file__).resolve().parents[1]


def test_desktop_window_uses_light_frameless_chrome() -> None:
    launcher = (APP_ROOT / "desktop_launcher.py").read_text(encoding="utf-8")
    assert 'APP_TITLE = "TOUCH"' in launcher
    assert 'app_url = f"http://127.0.0.1:{port}/?desktop=1"' in launcher
    assert "frameless=True" in launcher
    assert "easy_drag=False" in launcher
    assert "shadow=True" in launcher
    assert "resizable=False" not in launcher
    assert "maximized=False" in launcher
    assert "window.events.maximized += desktop_api.note_window_maximized" in launcher
    assert "window.events.restored += desktop_api.note_window_restored" in launcher


def test_custom_titlebar_is_nameless_and_keeps_native_window_commands() -> None:
    html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "desktop-titlebar-drag pywebview-drag-region" in html
    assert 'id="desktopMinimizeButton"' in html
    assert 'id="desktopMaximizeButton"' in html
    assert 'id="desktopCloseButton"' in html
    assert 'src="/static/touch_system_icon.png"' in html
    assert "desktop-titlebar-grip" not in html
    titlebar = html.split('id="desktopTitlebar"', 1)[1].split('<div class="app-shell', 1)[0]
    assert "TOUCH" not in titlebar
    assert "System" not in titlebar
    for command in ("minimize_window", "toggle_maximize_window", "close_window"):
        assert command in app_js


def test_product_name_is_touch_only_in_window_and_primary_brand() -> None:
    html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    spec = (APP_ROOT / "desktop_launcher.spec").read_text(encoding="utf-8")
    assert "<title>TOUCH</title>" in html
    brand = html.split('class="touch-brand-title"', 1)[1].split("</h1>", 1)[0]
    assert brand.count("TOUCH") == 1
    assert "System" not in brand
    assert "touch-brand-tag" not in brand
    assert 'name="TOUCH"' in spec
    assert 'name="TOUCH System - Trained Static Spectrum Twin"' not in spec


def test_desktop_chrome_is_light_and_leaves_surface_fullscreen_clean() -> None:
    css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
    chrome = css.split("/* Native desktop shell:", 1)[1].split(
        "/* Immersive thumb tactile surface:", 1
    )[0]
    assert "body.pywebview-desktop .desktop-titlebar" in chrome
    assert "#d6e2eb" in chrome
    assert "rgba(255, 255, 255, 0.99)" in chrome
    assert "#000" not in chrome.lower()
    assert "body.pywebview-desktop.surface-fullscreen-document .desktop-titlebar" in chrome
    assert "display: none" in chrome
    assert "desktop-titlebar-grip" not in css


def test_window_resize_defers_expensive_canvas_reflow() -> None:
    app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    resize_handler = app_js.split('window.addEventListener("resize"', 1)[1].split(
        "function handlePageVisibilityChange", 1
    )[0]
    assert "windowResizeActive = true" in resize_handler
    assert "window.setTimeout" in resize_handler
    assert "resizeThree()" in resize_handler
    assert "updateUI(state.frame)" not in resize_handler


def test_desktop_window_commands_call_the_native_window() -> None:
    window = Mock()
    api = DesktopApi(initially_maximized=True)
    with patch("desktop_launcher.webview.windows", [window]):
        assert api.minimize_window()["ok"] is True
        restored = api.toggle_maximize_window()
        maximized = api.toggle_maximize_window()
        assert api.close_window()["ok"] is True

    window.minimize.assert_called_once_with()
    assert window.toggle_fullscreen.call_count == 2
    window.restore.assert_not_called()
    window.maximize.assert_not_called()
    window.destroy.assert_called_once_with()
    assert restored == {
        "ok": True,
        "status": "window_restored",
        "maximized": False,
    }
    assert maximized == {
        "ok": True,
        "status": "window_maximized",
        "maximized": True,
    }


def test_native_window_events_keep_the_toggle_state_synchronized() -> None:
    window = Mock()
    api = DesktopApi(initially_maximized=True)
    api.note_window_restored()
    with patch("desktop_launcher.webview.windows", [window]):
        result = api.toggle_maximize_window()
    window.toggle_fullscreen.assert_called_once_with()
    assert result["maximized"] is True


def test_toggle_is_stable_when_native_events_fire_synchronously() -> None:
    api = DesktopApi(initially_maximized=True)
    window = Mock()
    window.toggle_fullscreen.side_effect = [
        api.note_window_restored,
        api.note_window_maximized,
    ]

    with patch("desktop_launcher.webview.windows", [window]):
        restored = api.toggle_maximize_window()
        maximized = api.toggle_maximize_window()

    assert restored == {
        "ok": True,
        "status": "window_restored",
        "maximized": False,
    }
    assert maximized == {
        "ok": True,
        "status": "window_maximized",
        "maximized": True,
    }
