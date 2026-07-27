from pathlib import Path
from unittest.mock import Mock, patch

from desktop_launcher import (
    GWL_STYLE,
    SWP_FRAMECHANGED,
    SWP_NOACTIVATE,
    SWP_NOMOVE,
    SWP_NOSIZE,
    SWP_NOZORDER,
    WS_MINIMIZEBOX,
    WS_SYSMENU,
    DesktopApi,
    enable_borderless_taskbar_toggle,
)
from desktop_window_zoom import (
    MAX_SNAPSHOT_HEIGHT,
    MAX_SNAPSHOT_WIDTH,
    ZoomRect,
    interpolate_zoom_rect,
    scaled_snapshot_size,
    smootherstep,
    thumbnail_rect_for_taskbar,
    union_zoom_rect,
)


APP_ROOT = Path(__file__).resolve().parents[1]


def test_desktop_window_uses_light_frameless_chrome() -> None:
    launcher = (APP_ROOT / "desktop_launcher.py").read_text(encoding="utf-8")
    assert "import ctypes" in launcher
    assert 'APP_TITLE = "TOUCH"' in launcher
    assert 'app_url = f"http://127.0.0.1:{port}/?desktop=1"' in launcher
    assert "frameless=True" in launcher
    assert "easy_drag=False" in launcher
    assert "shadow=True" in launcher
    assert "resizable=False" not in launcher
    assert "maximized=False" in launcher
    assert "window.events.maximized += desktop_api.note_window_maximized" in launcher
    assert "window.events.restored += desktop_api.note_window_restored" in launcher
    assert "window.events.shown += on_shown" in launcher
    assert "desktop_api.attach_window(window)" in launcher
    assert "html=startup_document(app_root)" in launcher
    assert "webview.start(finish_startup, debug=False)" in launcher


def test_borderless_window_restores_native_taskbar_toggle_contract() -> None:
    window = Mock()
    window.native.Handle.ToInt64.return_value = 1234
    user32 = Mock()
    user32.GetWindowLongPtrW.return_value = 0x10000000

    assert enable_borderless_taskbar_toggle(window, user32=user32) is True

    expected_style = 0x10000000 | WS_SYSMENU | WS_MINIMIZEBOX
    user32.SetWindowLongPtrW.assert_called_once_with(
        1234,
        GWL_STYLE,
        expected_style,
    )
    user32.SetWindowPos.assert_called_once_with(
        1234,
        0,
        0,
        0,
        0,
        0,
        SWP_NOSIZE
        | SWP_NOMOVE
        | SWP_NOZORDER
        | SWP_NOACTIVATE
        | SWP_FRAMECHANGED,
    )


def test_taskbar_toggle_setup_is_idempotent() -> None:
    window = Mock()
    window.native.Handle.ToInt64.return_value = 4321
    user32 = Mock()
    user32.GetWindowLongPtrW.return_value = WS_SYSMENU | WS_MINIMIZEBOX

    assert enable_borderless_taskbar_toggle(window, user32=user32) is True

    user32.SetWindowLongPtrW.assert_not_called()
    user32.SetWindowPos.assert_not_called()


def test_taskbar_thumbnail_preserves_the_window_aspect_ratio() -> None:
    source = ZoomRect(0.0, 0.0, 1280.0, 800.0)
    taskbar_button = ZoomRect(1633.0, 1504.0, 88.0, 96.0)
    target = thumbnail_rect_for_taskbar(source, taskbar_button)

    assert target.width / target.height == source.width / source.height
    assert target.center_x == taskbar_button.center_x
    assert target.center_y == taskbar_button.center_y


def test_snapshot_zoom_uses_zero_velocity_endpoints() -> None:
    source = ZoomRect(0.0, 0.0, 1280.0, 800.0)
    target = ZoomRect(1646.0, 1532.0, 62.0, 39.0)

    assert smootherstep(0.0) == 0.0
    assert smootherstep(1.0) == 1.0
    assert interpolate_zoom_rect(source, target, 0.0) == source
    assert interpolate_zoom_rect(source, target, 1.0) == target


def test_snapshot_zoom_caps_the_animated_bitmap_and_uses_fixed_overlay_bounds() -> None:
    width, height = scaled_snapshot_size(2048, 1280)
    source = ZoomRect(100.0, 80.0, 1200.0, 800.0)
    target = ZoomRect(900.0, 1040.0, 80.0, 52.0)
    overlay = union_zoom_rect(source, target)

    assert (width, height) == (896, 560)
    assert width <= MAX_SNAPSHOT_WIDTH
    assert height <= MAX_SNAPSHOT_HEIGHT
    assert overlay.x < source.x
    assert overlay.y < source.y
    assert overlay.x + overlay.width > target.x + target.width
    assert overlay.y + overlay.height > target.y + target.height


def test_custom_titlebar_carries_touch_and_keeps_native_window_commands() -> None:
    html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "desktop-titlebar-drag pywebview-drag-region" in html
    assert 'id="desktopMinimizeButton"' in html
    assert 'id="desktopMaximizeButton"' in html
    assert 'id="desktopCloseButton"' in html
    assert 'src="/static/touch_system_icon.png"' in html
    assert '<span class="desktop-titlebar-name">TOUCH</span>' in html
    assert "desktop-titlebar-grip" not in html
    titlebar = html.split('id="desktopTitlebar"', 1)[1].split('<div class="app-shell', 1)[0]
    assert titlebar.count("TOUCH") == 1
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


def test_desktop_workspace_moves_statuses_into_the_former_brand_column() -> None:
    css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
    assert "body.pywebview-desktop .desktop-titlebar-name" in css
    assert (
        "body.pywebview-desktop .app-shell:not(.surface-fullscreen-active) .title-zone"
        in css
    )
    assert (
        "grid-template-columns: minmax(360px, 430px) minmax(252px, 1fr)"
        in css
    )
    assert "justify-content: flex-start !important" in css


def test_desktop_workspace_reserves_titlebar_height_without_bottom_overflow() -> None:
    css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
    guard = css.split("/* Desktop viewport closure guard.", 1)[1]
    app_shell = guard.split(
        "body.pywebview-desktop .app-shell:not(.surface-fullscreen-active) {",
        1,
    )[1].split("}", 1)[0]

    assert "position: fixed !important" in app_shell
    assert "inset: var(--desktop-titlebar-height) 0 0 0 !important" in app_shell
    assert "height: auto !important" in app_shell
    assert "margin: 0 !important" in app_shell
    assert "overflow: hidden !important" in app_shell
    assert (
        "body.pywebview-desktop .app-shell:not(.surface-fullscreen-active) .right-panel"
        in guard
    )
    assert "max-height: 100% !important" in guard


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
    api._window_zoom = Mock()
    api._window_zoom.minimize.return_value = False
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
    api._window_zoom.detach.assert_called_once_with()
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


def test_minimize_uses_snapshot_transition_without_scaling_the_webgl_workspace() -> None:
    launcher = (APP_ROOT / "desktop_launcher.py").read_text(encoding="utf-8")
    zoom = (APP_ROOT / "desktop_window_zoom.py").read_text(encoding="utf-8")
    app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
    minimize_handler = app_js.split(
        'desktopMinimizeButton?.addEventListener("click"',
        1,
    )[1].split("desktopMaximizeButton?.addEventListener", 1)[0]
    assert 'invokeDesktopWindowCommand("minimize_window")' in minimize_handler
    assert "setTimeout" not in minimize_handler
    assert "DESKTOP_MINIMIZE_ANIMATION_MS" not in app_js
    assert "__touchDesktopWindowState" not in app_js
    assert "desktop-window-minimizing" not in css
    assert "desktop-window-restoring" not in css
    assert "touch-desktop-window-restore" not in css
    assert "MacStyleTaskbarZoom" in launcher
    assert "CopyFromScreen" in zoom
    assert "Taskbar.TaskListButtonAutomationPeer" in zoom
    assert "WM_SYSCOMMAND" in zoom
    assert "SC_MINIMIZE" in zoom
    assert "SC_RESTORE" in zoom
    assert "scaled_snapshot_size" in zoom
    assert 'Assembly.LoadFile(str(wpf_root / assembly_name))' in zoom
    assert '"DoubleAnimation": DoubleAnimation' in zoom
    assert "BeginAnimation(" in zoom
    assert '"ScaleTransform": ScaleTransform' in zoom
    assert '"TranslateTransform": TranslateTransform' in zoom
    transition = zoom.split("def _start_transition", 1)[1].split(
        "def _complete_minimize",
        1,
    )[0]
    assert "self._overlay.Opacity =" not in transition
    assert "self._overlay.Bounds =" not in transition
    assert "timer.Interval" not in transition
    assert "ShowWindowAsync" not in launcher


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
