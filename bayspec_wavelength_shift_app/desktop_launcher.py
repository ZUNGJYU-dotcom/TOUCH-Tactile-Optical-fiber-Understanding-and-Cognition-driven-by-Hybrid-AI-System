"""Desktop launcher for the TOUCH temporal spectral validation twin."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
import urllib.request

import uvicorn
import webview


APP_TITLE = "TOUCH System - Temporal Spectral Validation"
DEFAULT_PORT = 8640
EXPECTED_BACKEND_APP = "TOUCH System Trained Static Spectrum Twin"
EXPECTED_BACKEND_MODE = "standalone_bayspec_trained_static_spectrum_twin"
EXPECTED_BACKEND_CONTRACT_VERSION = "trained_static_spectrum_api_v2"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
    return Path(__file__).resolve().parent


def log_path() -> Path:
    base = (
        Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
        / "TouchSystemTrainedStaticSpectrumTwin"
        / "logs"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base / "desktop_launcher.log"


def write_log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path().open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def show_error(title: str, message: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, title, 0x00000010)
    except Exception:
        write_log(f"{title}: {message}")


def configure_runtime_paths() -> Path:
    app_root = bundle_root()
    os.environ["BAYSPEC_WAVELENGTH_APP_ROOT"] = str(app_root)
    if is_frozen():
        # Keep experiment outputs beside the portable desktop package instead
        # of burying them in PyInstaller's internal resource directory.
        portable_data_root = Path(sys.executable).resolve().parent / "data" / "px6d_synchronized"
        os.environ.setdefault("TOUCH_CAPTURE_OUTPUT_ROOT", str(portable_data_root))
    runtime_roots = [app_root]
    if not is_frozen():
        # Source launches import the shared recognition code from the project
        # level ``src`` package. Frozen builds bundle that package beside the
        # launcher and therefore continue to use app_root only.
        runtime_roots.append(app_root.parent)
    for runtime_root in reversed(runtime_roots):
        if str(runtime_root) not in sys.path:
            sys.path.insert(0, str(runtime_root))
    return app_root


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def require_fixed_port(port: int) -> None:
    if port_is_free(port):
        return
    message = (
        f"Port {port} is already in use.\n\n"
        "Another TOUCH System trained static-spectrum backend is already running. "
        "Close the old trained-spectrum app instance or stop its Python/uvicorn process, "
        "then start this app again. The wavelength-shift and optical-intensity editions use different ports."
    )
    write_log(message)
    show_error(APP_TITLE, message)
    raise RuntimeError(message)


def health_payload_is_expected(payload: object) -> bool:
    return bool(
        isinstance(payload, dict)
        and payload.get("ok") is True
        and payload.get("app") == EXPECTED_BACKEND_APP
        and payload.get("mode") == EXPECTED_BACKEND_MODE
        and payload.get("backend_contract_version")
        == EXPECTED_BACKEND_CONTRACT_VERSION
        and payload.get("trained_static_model_primary") is True
    )


def run_self_test() -> int:
    """Validate frozen resources without opening hardware, a port, or a window."""

    os.environ["TOUCH_PX6D_AUTO_START"] = "false"
    app_root = configure_runtime_paths()
    checks: dict[str, object] = {
        "app_root": str(app_root),
        "frozen": is_frozen(),
        "frontend_index": (app_root / "frontend" / "index.html").is_file(),
        "frontend_javascript": (app_root / "frontend" / "app.js").is_file(),
        "sdk_helper": (app_root / "sdk_probe" / "BaySpecSdkStream.exe").is_file(),
    }
    try:
        from backend.main import health

        payload = health()
        checks["backend_contract"] = health_payload_is_expected(payload)
        checks["static_model_loaded"] = bool(
            payload.get("trained_static_spectral_model", {}).get("loaded")
        )
        checks["dynamic_model_loaded"] = bool(
            payload.get("dynamic_temporal_shadow", {}).get("loaded")
        )
    except Exception as exc:
        checks["backend_import_error"] = f"{type(exc).__name__}: {exc}"
    ok = all(value is True for key, value in checks.items() if key not in {"app_root", "frozen"})
    write_log("SELF_TEST " + json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False))
    return 0 if ok else 1


def _read_expected_health(url: str, timeout_s: float) -> bool:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        if response.status != 200:
            return False
        payload = json.loads(response.read().decode("utf-8"))
        return health_payload_is_expected(payload)


def backend_is_ready(url: str, timeout_s: float = 0.8) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if _read_expected_health(url, timeout_s=0.4):
                return True
        except Exception:
            time.sleep(0.12)
    return False


def wait_until_ready(url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            if _read_expected_health(url, timeout_s=1.0):
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.15)
    raise RuntimeError(f"Backend did not become ready: {last_error}")


def run_backend(port: int, server_holder: dict[str, uvicorn.Server]) -> None:
    try:
        from backend.main import app

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        server_holder["server"] = server
        server.run()
    except Exception:
        write_log(traceback.format_exc())
        raise


class DesktopApi:
    """Small native-only API exposed to the bundled pywebview frontend."""

    def choose_output_directory(self, current_path: str = "") -> dict[str, object]:
        try:
            initial = Path(str(current_path or "")).expanduser()
            if not initial.is_dir():
                initial = initial.parent if initial.parent.is_dir() else Path.home()
            if not webview.windows:
                return {"ok": False, "status": "desktop_window_not_ready"}
            selected = webview.windows[0].create_file_dialog(
                webview.FileDialog.FOLDER,
                directory=str(initial),
            )
            if not selected:
                return {"ok": False, "status": "folder_selection_cancelled"}
            path = selected[0] if isinstance(selected, (list, tuple)) else selected
            return {"ok": True, "status": "folder_selected", "path": str(Path(path))}
        except Exception as exc:
            write_log(f"Output folder selection failed: {type(exc).__name__}: {exc}")
            return {
                "ok": False,
                "status": "folder_selection_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }


def main() -> int:
    app_root = configure_runtime_paths()
    write_log(f"Starting {APP_TITLE}; app_root={app_root}")
    port = DEFAULT_PORT
    server_holder: dict[str, uvicorn.Server] = {}
    health_url = f"http://127.0.0.1:{port}/api/health"
    app_url = f"http://127.0.0.1:{port}/"
    owns_backend = False
    if port_is_free(port):
        thread = threading.Thread(target=run_backend, args=(port, server_holder), daemon=True)
        thread.start()
        owns_backend = True
        wait_until_ready(health_url)
    elif backend_is_ready(health_url, timeout_s=1.2):
        write_log(f"Reusing existing trained static-spectrum backend on port {port}")
    else:
        require_fixed_port(port)

    window = webview.create_window(
        APP_TITLE,
        app_url,
        width=1180,
        height=760,
        min_size=(1024, 680),
        maximized=True,
        background_color="#f5f9fc",
        js_api=DesktopApi(),
    )

    def on_closed() -> None:
        if not owns_backend:
            return
        server = server_holder.get("server")
        if server is not None:
            server.should_exit = True

    window.events.closed += on_closed
    webview.start(debug=False)
    on_closed()
    return 0


if __name__ == "__main__":
    try:
        if "--self-test" in sys.argv:
            raise SystemExit(run_self_test())
        raise SystemExit(main())
    except Exception:
        write_log(traceback.format_exc())
        raise
