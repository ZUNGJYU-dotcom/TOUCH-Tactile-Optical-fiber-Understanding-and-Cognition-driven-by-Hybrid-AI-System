"""Native Windows snapshot animation for the frameless TOUCH desktop window."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import threading
from typing import Any, Callable


GWL_WNDPROC = -4
GWL_EXSTYLE = -20
WM_SYSCOMMAND = 0x0112
SC_MASK = 0xFFF0
SC_MINIMIZE = 0xF020
SC_RESTORE = 0xF120
SW_MINIMIZE = 6
SW_RESTORE = 9
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
MAX_SNAPSHOT_WIDTH = 900
MAX_SNAPSHOT_HEIGHT = 560


@dataclass(frozen=True)
class ZoomRect:
    """Screen-space rectangle used by the snapshot transition."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


def smootherstep(progress: float) -> float:
    """Return a zero-velocity ease at both transition endpoints."""

    value = max(0.0, min(1.0, float(progress)))
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def interpolate_zoom_rect(start: ZoomRect, end: ZoomRect, progress: float) -> ZoomRect:
    eased = smootherstep(progress)

    def blend(first: float, second: float) -> float:
        return first + (second - first) * eased

    return ZoomRect(
        x=blend(start.x, end.x),
        y=blend(start.y, end.y),
        width=max(1.0, blend(start.width, end.width)),
        height=max(1.0, blend(start.height, end.height)),
    )


def thumbnail_rect_for_taskbar(
    source: ZoomRect,
    taskbar_button: ZoomRect,
    *,
    occupancy: float = 0.70,
) -> ZoomRect:
    """Fit a small source-aspect thumbnail inside the actual taskbar button."""

    source_width = max(1.0, source.width)
    source_height = max(1.0, source.height)
    available_width = max(20.0, taskbar_button.width * occupancy)
    available_height = max(16.0, taskbar_button.height * occupancy)
    scale = min(available_width / source_width, available_height / source_height)
    width = max(20.0, source_width * scale)
    height = max(12.0, source_height * scale)
    return ZoomRect(
        x=taskbar_button.center_x - width / 2.0,
        y=taskbar_button.center_y - height / 2.0,
        width=width,
        height=height,
    )


def scaled_snapshot_size(
    width: float,
    height: float,
    *,
    max_width: int = MAX_SNAPSHOT_WIDTH,
    max_height: int = MAX_SNAPSHOT_HEIGHT,
) -> tuple[int, int]:
    """Return a compact aspect-preserving snapshot size for animated scaling."""

    source_width = max(1, int(round(width)))
    source_height = max(1, int(round(height)))
    scale = min(
        1.0,
        max(1, int(max_width)) / source_width,
        max(1, int(max_height)) / source_height,
    )
    return (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )


def union_zoom_rect(
    first: ZoomRect,
    second: ZoomRect,
    *,
    padding: float = 3.0,
) -> ZoomRect:
    """Return a fixed overlay rectangle containing both transition endpoints."""

    margin = max(0.0, float(padding))
    left = min(first.x, second.x) - margin
    top = min(first.y, second.y) - margin
    right = max(first.x + first.width, second.x + second.width) + margin
    bottom = max(first.y + first.height, second.y + second.height) + margin
    return ZoomRect(left, top, max(1.0, right - left), max(1.0, bottom - top))


class MacStyleTaskbarZoom:
    """Animate a window snapshot to and from its Windows taskbar button."""

    def __init__(
        self,
        *,
        app_title: str,
        logger: Callable[[str], None] | None = None,
        minimize_duration_s: float = 0.28,
        restore_duration_s: float = 0.30,
    ) -> None:
        self.app_title = str(app_title)
        self._log = logger or (lambda _message: None)
        self.minimize_duration_s = max(0.12, float(minimize_duration_s))
        self.restore_duration_s = max(0.12, float(restore_duration_s))
        self._window: Any | None = None
        self._native: Any | None = None
        self._hwnd = 0
        self._attached = False
        self._animating = False
        self._is_minimized = False
        self._internal_window_change = False
        self._restore_rect: ZoomRect | None = None
        self._snapshot: Any | None = None
        self._overlay: Any | None = None
        self._overlay_scale: Any | None = None
        self._overlay_translate: Any | None = None
        self._overlay_image_source: Any | None = None
        self._animations: list[Any] = []
        self._animation_callback: Any | None = None
        self._taskbar_button_rect: ZoomRect | None = None
        self._wndproc_callback: Any | None = None
        self._previous_wndproc = 0
        self._dotnet_ready = False
        self._dotnet: dict[str, Any] = {}
        self._user32: Any | None = None
        self._gdi32: Any | None = None
        self._callback_type: Any | None = None

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def animating(self) -> bool:
        return self._animating

    def attach(self, window: Any) -> bool:
        if os.name != "nt":
            return False
        native = getattr(window, "native", None)
        handle = getattr(native, "Handle", None)
        if native is None or handle is None:
            return False
        try:
            self._ensure_dotnet()
            hwnd = int(handle.ToInt64()) if hasattr(handle, "ToInt64") else int(handle)
            if hwnd <= 0:
                return False
            self._window = window
            self._native = native
            self._hwnd = hwnd
            self._run_on_ui(self._install_window_hook, synchronous=True)
            self._attached = self._previous_wndproc != 0
            if self._attached:
                threading.Thread(
                    target=self._refresh_taskbar_button_rect,
                    name="touch-taskbar-target-discovery",
                    daemon=True,
                ).start()
                self._log(f"Desktop snapshot zoom attached hwnd={self._hwnd}")
            return self._attached
        except Exception as exc:
            self._log(
                "Desktop snapshot zoom attach failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

    def detach(self) -> None:
        if not self._attached:
            self._dispose_visuals()
            return
        try:
            self._run_on_ui(self._detach_on_ui, synchronous=True)
        except Exception as exc:
            self._log(
                "Desktop snapshot zoom detach failed: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            self._attached = False

    def minimize(self, window: Any | None = None) -> bool:
        if not self._attached and window is not None and not self.attach(window):
            return False
        if not self._attached or self._animating:
            return False
        return self._run_on_ui(self._start_minimize, synchronous=False)

    def restore(self) -> bool:
        if not self._attached or self._animating or not self._is_minimized:
            return False
        return self._run_on_ui(self._start_restore, synchronous=False)

    def note_restored(self) -> None:
        if not self._internal_window_change and not self._animating:
            self._is_minimized = False

    def _ensure_dotnet(self) -> None:
        if self._dotnet_ready:
            return
        import clr

        clr.AddReference("System")
        clr.AddReference("System.Drawing")
        clr.AddReference("System.Windows.Forms")
        from System import Action, IntPtr, TimeSpan
        from System.Drawing import Bitmap, Graphics, Rectangle, Size
        from System.Drawing.Drawing2D import InterpolationMode
        from System.Reflection import Assembly

        windows_root = Path(os.environ.get("WINDIR", r"C:\Windows"))
        framework_name = "Framework64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "Framework"
        wpf_root = windows_root / "Microsoft.NET" / framework_name / "v4.0.30319" / "WPF"
        for assembly_name in (
            "WindowsBase.dll",
            "PresentationCore.dll",
            "PresentationFramework.dll",
        ):
            Assembly.LoadFile(str(wpf_root / assembly_name))

        from System.Windows import (
            Duration,
            Int32Rect,
            ResizeMode,
            Window,
            WindowStyle,
        )
        from System.Windows.Controls import Canvas, Image
        from System.Windows.Interop import Imaging, WindowInteropHelper
        from System.Windows.Media import (
            BitmapCache,
            BitmapScalingMode,
            Brushes,
            RenderOptions,
            ScaleTransform,
            Stretch,
            TranslateTransform,
        )
        from System.Windows.Media.Animation import (
            DoubleAnimation,
            EasingMode,
            FillBehavior,
            QuinticEase,
        )
        from System.Windows.Media.Imaging import BitmapSizeOptions

        self._dotnet = {
            "Action": Action,
            "IntPtr": IntPtr,
            "TimeSpan": TimeSpan,
            "Bitmap": Bitmap,
            "Graphics": Graphics,
            "Rectangle": Rectangle,
            "Size": Size,
            "InterpolationMode": InterpolationMode,
            "Duration": Duration,
            "Int32Rect": Int32Rect,
            "ResizeMode": ResizeMode,
            "Window": Window,
            "WindowStyle": WindowStyle,
            "Canvas": Canvas,
            "Image": Image,
            "Imaging": Imaging,
            "WindowInteropHelper": WindowInteropHelper,
            "BitmapCache": BitmapCache,
            "BitmapScalingMode": BitmapScalingMode,
            "Brushes": Brushes,
            "RenderOptions": RenderOptions,
            "ScaleTransform": ScaleTransform,
            "Stretch": Stretch,
            "TranslateTransform": TranslateTransform,
            "DoubleAnimation": DoubleAnimation,
            "EasingMode": EasingMode,
            "FillBehavior": FillBehavior,
            "QuinticEase": QuinticEase,
            "BitmapSizeOptions": BitmapSizeOptions,
        }
        self._dotnet_ready = True

    def _run_on_ui(self, callback: Callable[[], None], *, synchronous: bool) -> bool:
        native = self._native
        if native is None:
            return False
        try:
            action = self._dotnet["Action"](callback)
            if not synchronous:
                native.BeginInvoke(action)
            elif bool(getattr(native, "InvokeRequired", False)):
                native.Invoke(action)
            else:
                callback()
            return True
        except Exception as exc:
            self._log(
                "Desktop snapshot zoom UI dispatch failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

    def _configure_user32(self) -> Any:
        if self._user32 is not None:
            return self._user32
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        pointer_type = ctypes.c_ssize_t
        user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.GetWindowLongPtrW.restype = pointer_type
        user32.SetWindowLongPtrW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            pointer_type,
        ]
        user32.SetWindowLongPtrW.restype = pointer_type
        user32.CallWindowProcW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_ssize_t,
        ]
        user32.CallWindowProcW.restype = pointer_type
        user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.ShowWindow.restype = ctypes.c_bool
        user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
        user32.SetForegroundWindow.restype = ctypes.c_bool
        user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        user32.GetWindowRect.restype = ctypes.c_bool
        user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user32.FindWindowW.restype = ctypes.c_void_p
        try:
            user32.GetDpiForWindow.argtypes = [ctypes.c_void_p]
            user32.GetDpiForWindow.restype = ctypes.c_uint
        except AttributeError:
            pass
        self._user32 = user32
        return user32

    def _configure_gdi32(self) -> Any:
        if self._gdi32 is not None:
            return self._gdi32
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
        gdi32.DeleteObject.restype = ctypes.c_bool
        self._gdi32 = gdi32
        return gdi32

    def _install_window_hook(self) -> None:
        if self._previous_wndproc:
            return
        user32 = self._configure_user32()
        callback_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_ssize_t,
        )
        self._callback_type = callback_type

        def window_proc(
            hwnd: int,
            message: int,
            wparam: int,
            lparam: int,
        ) -> int:
            try:
                command = int(wparam) & SC_MASK
                if message == WM_SYSCOMMAND and not self._internal_window_change:
                    if self._animating and command in (SC_MINIMIZE, SC_RESTORE):
                        return 0
                    if command == SC_MINIMIZE:
                        self.minimize()
                        return 0
                    if command == SC_RESTORE and self._is_minimized:
                        self.restore()
                        return 0
            except Exception as exc:
                self._log(
                    "Desktop snapshot zoom window hook error: "
                    f"{type(exc).__name__}: {exc}"
                )
            return int(
                user32.CallWindowProcW(
                    self._previous_wndproc,
                    hwnd,
                    message,
                    wparam,
                    lparam,
                )
            )

        self._wndproc_callback = callback_type(window_proc)
        callback_address = ctypes.cast(
            self._wndproc_callback,
            ctypes.c_void_p,
        ).value
        ctypes.set_last_error(0)
        previous = int(
            user32.SetWindowLongPtrW(
                self._hwnd,
                GWL_WNDPROC,
                int(callback_address or 0),
            )
        )
        if previous == 0 and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        self._previous_wndproc = previous

    def _detach_on_ui(self) -> None:
        self._dispose_visuals()
        if self._previous_wndproc and self._hwnd:
            user32 = self._configure_user32()
            user32.SetWindowLongPtrW(
                self._hwnd,
                GWL_WNDPROC,
                self._previous_wndproc,
            )
        self._previous_wndproc = 0
        self._wndproc_callback = None

    def _window_rect(self, hwnd: int) -> ZoomRect | None:
        class Rect(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        rect = Rect()
        if not self._configure_user32().GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width < 2 or height < 2 or rect.left < -10000 or rect.top < -10000:
            return None
        return ZoomRect(float(rect.left), float(rect.top), float(width), float(height))

    def _capture_window(self, rect: ZoomRect) -> Any:
        source_width = max(1, int(round(rect.width)))
        source_height = max(1, int(round(rect.height)))
        source_bitmap = self._dotnet["Bitmap"](source_width, source_height)
        graphics = self._dotnet["Graphics"].FromImage(source_bitmap)
        try:
            graphics.CopyFromScreen(
                int(round(rect.x)),
                int(round(rect.y)),
                0,
                0,
                self._dotnet["Size"](source_width, source_height),
            )
        finally:
            graphics.Dispose()
        snapshot_width, snapshot_height = scaled_snapshot_size(
            source_width,
            source_height,
        )
        if (snapshot_width, snapshot_height) == (source_width, source_height):
            return source_bitmap

        snapshot = self._dotnet["Bitmap"](snapshot_width, snapshot_height)
        graphics = self._dotnet["Graphics"].FromImage(snapshot)
        try:
            graphics.InterpolationMode = self._dotnet[
                "InterpolationMode"
            ].HighQualityBilinear
            graphics.DrawImage(
                source_bitmap,
                0,
                0,
                snapshot_width,
                snapshot_height,
            )
        finally:
            graphics.Dispose()
            source_bitmap.Dispose()
        return snapshot

    def _window_dip_scale(self) -> float:
        getter = getattr(self._configure_user32(), "GetDpiForWindow", None)
        if getter is None:
            return 1.0
        try:
            dpi = max(1, int(getter(self._hwnd)))
            return 96.0 / float(dpi)
        except Exception:
            return 1.0

    def _bitmap_source(self, bitmap: Any) -> Any:
        handle = bitmap.GetHbitmap()
        handle_value = (
            int(handle.ToInt64()) if hasattr(handle, "ToInt64") else int(handle)
        )
        try:
            source = self._dotnet["Imaging"].CreateBitmapSourceFromHBitmap(
                handle,
                self._dotnet["IntPtr"].Zero,
                self._dotnet["Int32Rect"].Empty,
                self._dotnet["BitmapSizeOptions"].FromEmptyOptions(),
            )
            source.Freeze()
            return source
        finally:
            self._configure_gdi32().DeleteObject(handle_value)

    def _create_overlay(
        self,
        bitmap: Any,
        start: ZoomRect,
        end: ZoomRect,
    ) -> Any:
        base = self._restore_rect or (
            start if start.width * start.height >= end.width * end.height else end
        )
        overlay_bounds = union_zoom_rect(start, end)
        dip_scale = self._window_dip_scale()

        overlay = self._dotnet["Window"]()
        overlay.WindowStyle = getattr(self._dotnet["WindowStyle"], "None")
        overlay.ResizeMode = self._dotnet["ResizeMode"].NoResize
        overlay.AllowsTransparency = True
        overlay.Background = self._dotnet["Brushes"].Transparent
        overlay.ShowInTaskbar = False
        overlay.ShowActivated = False
        overlay.Topmost = True
        overlay.Focusable = False
        overlay.IsHitTestVisible = False
        overlay.Left = overlay_bounds.x * dip_scale
        overlay.Top = overlay_bounds.y * dip_scale
        overlay.Width = max(1.0, overlay_bounds.width * dip_scale)
        overlay.Height = max(1.0, overlay_bounds.height * dip_scale)

        canvas = self._dotnet["Canvas"]()
        canvas.Width = overlay.Width
        canvas.Height = overlay.Height
        canvas.ClipToBounds = True
        canvas.IsHitTestVisible = False

        layer = self._dotnet["Canvas"]()
        layer.Width = max(1.0, base.width * dip_scale)
        layer.Height = max(1.0, base.height * dip_scale)
        layer.IsHitTestVisible = False
        self._dotnet["Canvas"].SetLeft(
            layer,
            (base.x - overlay_bounds.x) * dip_scale,
        )
        self._dotnet["Canvas"].SetTop(
            layer,
            (base.y - overlay_bounds.y) * dip_scale,
        )

        image = self._dotnet["Image"]()
        image.Width = layer.Width
        image.Height = layer.Height
        image.Stretch = self._dotnet["Stretch"].Fill
        image.IsHitTestVisible = False
        image_source = self._bitmap_source(bitmap)
        image.Source = image_source
        self._dotnet["RenderOptions"].SetBitmapScalingMode(
            image,
            self._dotnet["BitmapScalingMode"].LowQuality,
        )
        image.CacheMode = self._dotnet["BitmapCache"](1.0)

        scale = self._dotnet["ScaleTransform"](
            start.width / max(1.0, base.width),
            start.height / max(1.0, base.height),
        )
        translate = self._dotnet["TranslateTransform"](
            (start.x - base.x) * dip_scale,
            (start.y - base.y) * dip_scale,
        )
        image.RenderTransform = scale
        layer.RenderTransform = translate
        layer.Children.Add(image)
        canvas.Children.Add(layer)
        overlay.Content = canvas
        overlay.Show()
        overlay.UpdateLayout()

        overlay_hwnd = self._dotnet["WindowInteropHelper"](overlay).Handle
        overlay_hwnd_value = (
            int(overlay_hwnd.ToInt64())
            if hasattr(overlay_hwnd, "ToInt64")
            else int(overlay_hwnd)
        )
        if overlay_hwnd_value:
            user32 = self._configure_user32()
            extended_style = int(
                user32.GetWindowLongPtrW(overlay_hwnd_value, GWL_EXSTYLE)
            )
            user32.SetWindowLongPtrW(
                overlay_hwnd_value,
                GWL_EXSTYLE,
                extended_style
                | WS_EX_TRANSPARENT
                | WS_EX_TOOLWINDOW
                | WS_EX_NOACTIVATE,
            )

        self._overlay_scale = scale
        self._overlay_translate = translate
        self._overlay_image_source = image_source
        return overlay

    def _start_minimize(self) -> None:
        if self._animating or self._is_minimized:
            return
        source = self._window_rect(self._hwnd)
        if source is None:
            return
        try:
            self._dispose_snapshot()
            self._snapshot = self._capture_window(source)
            self._restore_rect = source
            target = self._taskbar_thumbnail_rect(source)
            self._overlay = self._create_overlay(self._snapshot, source, target)
            self._internal_show_window(SW_MINIMIZE)
            self._start_transition(
                source,
                target,
                duration_s=self.minimize_duration_s,
                restoring=False,
                on_complete=self._complete_minimize,
            )
        except Exception as exc:
            self._log(
                "Desktop snapshot minimize failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self._dispose_visuals()
            self._internal_show_window(SW_MINIMIZE)
            self._is_minimized = True

    def _start_restore(self) -> None:
        if self._animating:
            return
        restore_rect = self._restore_rect
        if restore_rect is None or self._snapshot is None:
            self._internal_show_window(SW_RESTORE)
            self._configure_user32().SetForegroundWindow(self._hwnd)
            self._is_minimized = False
            return
        try:
            source = self._taskbar_thumbnail_rect(restore_rect)
            self._overlay = self._create_overlay(
                self._snapshot,
                source,
                restore_rect,
            )
            self._start_transition(
                source,
                restore_rect,
                duration_s=self.restore_duration_s,
                restoring=True,
                on_complete=self._complete_restore,
            )
        except Exception as exc:
            self._log(
                "Desktop snapshot restore failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self._dispose_overlay()
            self._internal_show_window(SW_RESTORE)
            self._configure_user32().SetForegroundWindow(self._hwnd)
            self._is_minimized = False

    def _start_transition(
        self,
        start: ZoomRect,
        end: ZoomRect,
        *,
        duration_s: float,
        restoring: bool,
        on_complete: Callable[[], None],
    ) -> None:
        self._animating = True
        base = self._restore_rect or (
            start if start.width * start.height >= end.width * end.height else end
        )
        dip_scale = self._window_dip_scale()
        duration = self._dotnet["Duration"](
            self._dotnet["TimeSpan"].FromSeconds(max(0.001, duration_s))
        )
        easing = self._dotnet["QuinticEase"]()
        easing.EasingMode = self._dotnet["EasingMode"].EaseInOut

        def animation(start_value: float, end_value: float) -> Any:
            instance = self._dotnet["DoubleAnimation"](
                float(start_value),
                float(end_value),
                duration,
            )
            instance.EasingFunction = easing
            instance.FillBehavior = self._dotnet["FillBehavior"].HoldEnd
            return instance

        scale_x_animation = animation(
            start.width / max(1.0, base.width),
            end.width / max(1.0, base.width),
        )
        scale_y_animation = animation(
            start.height / max(1.0, base.height),
            end.height / max(1.0, base.height),
        )
        translate_x_animation = animation(
            (start.x - base.x) * dip_scale,
            (end.x - base.x) * dip_scale,
        )
        translate_y_animation = animation(
            (start.y - base.y) * dip_scale,
            (end.y - base.y) * dip_scale,
        )

        def on_finished(_sender: Any, _event: Any) -> None:
            try:
                translate_y_animation.Completed -= on_finished
            except Exception:
                pass
            self._animations = []
            self._animation_callback = None
            self._animating = False
            on_complete()

        self._animations = [
            scale_x_animation,
            scale_y_animation,
            translate_x_animation,
            translate_y_animation,
        ]
        self._animation_callback = on_finished
        translate_y_animation.Completed += on_finished

        if self._overlay_scale is None or self._overlay_translate is None:
            raise RuntimeError("snapshot compositor was not initialized")
        self._overlay_scale.BeginAnimation(
            self._dotnet["ScaleTransform"].ScaleXProperty,
            scale_x_animation,
        )
        self._overlay_scale.BeginAnimation(
            self._dotnet["ScaleTransform"].ScaleYProperty,
            scale_y_animation,
        )
        self._overlay_translate.BeginAnimation(
            self._dotnet["TranslateTransform"].XProperty,
            translate_x_animation,
        )
        self._overlay_translate.BeginAnimation(
            self._dotnet["TranslateTransform"].YProperty,
            translate_y_animation,
        )

    def _complete_minimize(self) -> None:
        self._dispose_overlay()
        self._is_minimized = True
        self._log("Desktop snapshot minimize animation complete")

    def _complete_restore(self) -> None:
        self._internal_show_window(SW_RESTORE)
        self._configure_user32().SetForegroundWindow(self._hwnd)
        self._dispose_overlay()
        self._dispose_snapshot()
        self._is_minimized = False
        self._log("Desktop snapshot restore animation complete")

    def _internal_show_window(self, command: int) -> None:
        self._internal_window_change = True
        try:
            self._configure_user32().ShowWindow(self._hwnd, int(command))
        finally:
            self._internal_window_change = False

    def _taskbar_thumbnail_rect(self, source: ZoomRect) -> ZoomRect:
        button = self._taskbar_button_rect
        if button is None:
            self._refresh_taskbar_button_rect()
            button = self._taskbar_button_rect
        if button is None:
            button = self._fallback_taskbar_rect(source)
        return thumbnail_rect_for_taskbar(source, button)

    def _refresh_taskbar_button_rect(self) -> None:
        try:
            rect = self._find_taskbar_button_rect()
            if rect is not None:
                self._taskbar_button_rect = rect
                self._log(
                    "Desktop taskbar target discovered "
                    f"x={rect.x:.0f} y={rect.y:.0f} "
                    f"w={rect.width:.0f} h={rect.height:.0f}"
                )
        except Exception as exc:
            self._log(
                "Desktop taskbar target discovery failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def _find_taskbar_button_rect(self) -> ZoomRect | None:
        from System import IntPtr
        from System.Reflection import Assembly

        windows_root = Path(os.environ.get("WINDIR", r"C:\Windows"))
        assembly_root = (
            windows_root
            / "Microsoft.NET"
            / "assembly"
            / "GAC_MSIL"
        )
        for name, folder in (
            ("UIAutomationTypes.dll", "UIAutomationTypes"),
            ("UIAutomationClient.dll", "UIAutomationClient"),
        ):
            matches = sorted((assembly_root / folder).glob(f"v4.0_*\\{name}"))
            if not matches:
                return None
            Assembly.LoadFile(str(matches[-1]))
        from System.Windows.Automation import (
            AutomationElement,
            Condition,
            ControlType,
            TreeScope,
        )

        taskbar_hwnd = self._configure_user32().FindWindowW("Shell_TrayWnd", None)
        if not taskbar_hwnd:
            return None
        root = AutomationElement.FromHandle(IntPtr(int(taskbar_hwnd)))
        descendants = root.FindAll(TreeScope.Descendants, Condition.TrueCondition)
        title = self.app_title.casefold()
        for index in range(descendants.Count):
            element = descendants[index]
            try:
                current = element.Current
                if current.ControlType != ControlType.Button:
                    continue
                if current.ClassName != "Taskbar.TaskListButtonAutomationPeer":
                    continue
                if not str(current.Name or "").casefold().startswith(title):
                    continue
                rect = current.BoundingRectangle
                if rect.Width > 1 and rect.Height > 1:
                    return ZoomRect(
                        float(rect.X),
                        float(rect.Y),
                        float(rect.Width),
                        float(rect.Height),
                    )
            except Exception:
                continue
        return None

    def _fallback_taskbar_rect(self, source: ZoomRect) -> ZoomRect:
        taskbar_hwnd = self._configure_user32().FindWindowW("Shell_TrayWnd", None)
        taskbar = self._window_rect(int(taskbar_hwnd or 0))
        if taskbar is None:
            return ZoomRect(
                source.center_x - 44.0,
                source.y + source.height - 96.0,
                88.0,
                96.0,
            )
        if taskbar.width >= taskbar.height:
            center_x = max(
                taskbar.x + 44.0,
                min(source.center_x, taskbar.x + taskbar.width - 44.0),
            )
            return ZoomRect(center_x - 44.0, taskbar.y, 88.0, taskbar.height)
        center_y = max(
            taskbar.y + 44.0,
            min(source.center_y, taskbar.y + taskbar.height - 44.0),
        )
        return ZoomRect(taskbar.x, center_y - 44.0, taskbar.width, 88.0)

    def _dispose_overlay(self) -> None:
        overlay = self._overlay
        self._overlay = None
        self._overlay_scale = None
        self._overlay_translate = None
        self._overlay_image_source = None
        self._animations = []
        self._animation_callback = None
        if overlay is None:
            return
        try:
            overlay.Close()
        except Exception:
            pass

    def _dispose_snapshot(self) -> None:
        snapshot = self._snapshot
        self._snapshot = None
        if snapshot is None:
            return
        try:
            snapshot.Dispose()
        except Exception:
            pass

    def _dispose_visuals(self) -> None:
        self._animating = False
        self._dispose_overlay()
        self._dispose_snapshot()
