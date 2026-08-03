"""Desktop launcher for the TOUCH temporal spectral validation twin."""

from __future__ import annotations

import base64
import ctypes
from html import escape
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from typing import Any
import urllib.request

import webview

from desktop_window_zoom import MacStyleTaskbarZoom


APP_TITLE = "TOUCH"
APP_EXPANDED_TITLE = (
    "Tactile Optical-fiber Understanding and Cognition driven by Hybrid-AI System"
)
DEFAULT_PORT = 8640
FALLBACK_PORT_COUNT = 10
EXPECTED_BACKEND_APP = "TOUCH System Trained Static Spectrum Twin"
EXPECTED_BACKEND_MODE = "standalone_bayspec_trained_static_spectrum_twin"
EXPECTED_BACKEND_CONTRACT_VERSION = "trained_static_spectrum_api_v2"
EXPECTED_OPERATOR_RECOGNITION = "dynamic_temporal_v3_validation"
EXPECTED_BETA_OPERATOR_RECOGNITION = "ordinary_fbg_all_data_beta_v1"
BETA_RUNTIME_FLAG_FILENAME = "beta_all_data_runtime.flag"
LATEST_RUNTIME_FLAG_FILENAME = "latest_all_data_runtime.flag"

GWL_STYLE = -16
WS_SYSMENU = 0x00080000
WS_MINIMIZEBOX = 0x00020000
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020


_FALLBACK_STARTUP_LOGO = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
<path fill="#78E2FE" d="M336 220C470 220 590 230 646 260C690 286 714 330 709 381C707 400 698 407 681 398C620 378 570 366 530 378C475 394 441 449 433 515C421 612 455 682 519 718C577 748 636 727 688 694C725 671 755 684 770 710C786 740 770 808 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z"/>
<circle cx="566" cy="551" r="181" fill="#075C73"/>
<circle cx="659" cy="552" r="149" fill="#FE9985"/>
</svg>"""

_ANIMATED_STARTUP_LOGO = """<svg class="contact-logo" xmlns="http://www.w3.org/2000/svg"
  viewBox="34 174 800 824" role="img" aria-label="TOUCH contact animation">
  <defs>
    <linearGradient id="startup-body" x1="0.16" y1="0.02" x2="0.78" y2="0.98">
      <stop offset="0" stop-color="#9AECFF"/>
      <stop offset="0.30" stop-color="#78E2FE"/>
      <stop offset="0.70" stop-color="#68DBF8"/>
      <stop offset="1" stop-color="#5BD3F3"/>
    </linearGradient>
    <radialGradient id="startup-body-light" cx="0.28" cy="0.18" r="0.88">
      <stop offset="0" stop-color="#FFFFFF" stop-opacity="0.34"/>
      <stop offset="0.45" stop-color="#FFFFFF" stop-opacity="0.08"/>
      <stop offset="1" stop-color="#0C8FAE" stop-opacity="0.10"/>
    </radialGradient>
    <radialGradient id="startup-cavity" cx="0.68" cy="0.46" r="0.72">
      <stop offset="0" stop-color="#0E7893"/>
      <stop offset="0.62" stop-color="#075C73"/>
      <stop offset="1" stop-color="#064B60"/>
    </radialGradient>
    <radialGradient id="startup-contact" cx="0.32" cy="0.24" r="0.86">
      <stop offset="0" stop-color="#FFB5A5"/>
      <stop offset="0.48" stop-color="#FE9985"/>
      <stop offset="1" stop-color="#EF6E59"/>
    </radialGradient>
    <filter id="startup-body-depth" x="-18%" y="-18%" width="136%" height="144%">
      <feDropShadow dx="0" dy="14" stdDeviation="13" flood-color="#0E7893" flood-opacity="0.20"/>
    </filter>
    <filter id="startup-ball-depth" x="-35%" y="-35%" width="170%" height="170%">
      <feDropShadow dx="-2" dy="9" stdDeviation="10" flood-color="#A64538" flood-opacity="0.22"/>
    </filter>
    <filter id="startup-cavity-depth" x="-35%" y="-35%" width="170%" height="170%">
      <feGaussianBlur stdDeviation="2.4"/>
      <feDropShadow dx="6" dy="4" stdDeviation="10" flood-color="#064B60" flood-opacity="0.34"/>
    </filter>
    <clipPath id="startup-outer-clip">
      <path d="M336 220C470 220 590 230 646 260C690 286 714 330 720 381C724 400 730 416 735 430C742 452 747 470 748 490C752 512 756 535 756 555C758 580 758 605 756 625C754 648 751 670 748 690C744 718 741 742 744 764C749 795 754 822 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z"/>
    </clipPath>
  </defs>
  <g class="soft-shell" filter="url(#startup-body-depth)">
    <path class="body-depth morph-shell"
      d="M336 220C470 220 590 230 646 260C690 286 714 330 720 381C724 400 730 416 735 430C742 452 747 470 748 490C752 512 756 535 756 555C758 580 758 605 756 625C754 648 751 670 748 690C744 718 741 742 744 764C749 795 754 822 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z"
      fill="#28B4D2" opacity="0.40" transform="translate(0 13)"/>
    <g clip-path="url(#startup-outer-clip)">
      <circle class="cavity-wall" cx="566" cy="551" r="181"
        fill="url(#startup-cavity)" filter="url(#startup-cavity-depth)"/>
    </g>
    <path class="body-shape morph-shell"
      d="M336 220C470 220 590 230 646 260C690 286 714 330 720 381C724 400 730 416 735 430C742 452 747 470 748 490C752 512 756 535 756 555C758 580 758 605 756 625C754 648 751 670 748 690C744 718 741 742 744 764C749 795 754 822 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z"
      fill="url(#startup-body)"/>
    <path class="body-light morph-shell"
      d="M336 220C470 220 590 230 646 260C690 286 714 330 720 381C724 400 730 416 735 430C742 452 747 470 748 490C752 512 756 535 756 555C758 580 758 605 756 625C754 648 751 670 748 690C744 718 741 742 744 764C749 795 754 822 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z"
      fill="url(#startup-body-light)" opacity="0.70"/>
    <path class="body-highlight" fill="none" stroke="#FFFFFF" stroke-opacity="0.24"
      stroke-width="7" stroke-linecap="round"
      d="M224 239C358 219 538 226 625 254C669 269 694 296 706 332"/>
  </g>
  <g class="contact-ball" filter="url(#startup-ball-depth)">
    <circle cx="659" cy="552" r="149" fill="url(#startup-contact)"/>
  </g>
</svg>"""


def startup_logo_data_uri(app_root: Path) -> str:
    """Embed the startup logo so WebView never depends on a local file URL."""

    logo_path = app_root / "frontend" / "touch_system_icon.png"
    try:
        logo_bytes = logo_path.read_bytes()
        media_type = "image/png"
    except OSError:
        logo_bytes = _FALLBACK_STARTUP_LOGO
        media_type = "image/svg+xml"
    encoded = base64.b64encode(logo_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def startup_document(app_root: Path, *, failed: bool = False) -> str:
    """Return the lightweight page shown while the backend initializes."""

    logo_uri = startup_logo_data_uri(app_root)
    title = "Unable to start" if failed else "Starting"
    detail = "Close TOUCH and try again" if failed else "Preparing workspace"
    state_class = "is-failed" if failed else "is-loading"
    activity = "" if failed else """
      <div class="progress" aria-hidden="true">
        <span></span>
      </div>
    """
    close_action = (
        """
      <button class="close-action" type="button"
        onclick="window.pywebview?.api?.close_window()">Close</button>
        """
        if failed
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{escape(APP_TITLE)}</title>
  <style>
    :root {{ color-scheme: light; font-family: "Segoe UI Variable", "Segoe UI", sans-serif; }}
    * {{ box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; }}
    body {{
      background:
        radial-gradient(circle at 50% 48%, rgba(92, 210, 234, 0.10), transparent 28%),
        #f4f7fa;
      color: #102236;
    }}
    .titlebar {{
      height: 48px; display: flex; align-items: center; gap: 10px;
      padding: 0 18px; border-bottom: 1px solid #d8e3ec;
      background: rgba(255, 255, 255, 0.94);
    }}
    .pywebview-drag-region {{ -webkit-app-region: drag; }}
    .titlebar img {{ width: 25px; height: 25px; object-fit: contain; }}
    .titlebar strong {{ font-size: 15px; letter-spacing: 0; }}
    main {{
      height: calc(100% - 48px); display: grid; place-items: center;
      padding: 32px;
    }}
    .startup {{
      width: min(620px, 86vw);
      display: grid;
      justify-items: center;
      gap: 12px;
      text-align: center;
    }}
    .logo-stage {{
      position: relative;
      width: 136px;
      height: 136px;
      display: grid;
      place-items: center;
      margin-bottom: 2px;
      isolation: isolate;
    }}
    .logo-stage::before,
    .logo-stage::after {{
      content: "";
      position: absolute;
      inset: 17px;
      border-radius: 42%;
      opacity: 0;
      pointer-events: none;
    }}
    .logo-stage::before {{
      border: 1px solid rgba(91, 211, 243, 0.30);
    }}
    .logo-stage::after {{
      inset: 23px;
      z-index: -1;
      background: radial-gradient(circle, rgba(120, 226, 254, 0.30), transparent 68%);
      filter: blur(8px);
    }}
    .logo-stage svg {{
      position: relative;
      z-index: 1;
      width: 124px;
      height: 124px;
      overflow: visible;
    }}
    .morph-shell, .soft-shell, .contact-ball, .cavity-wall {{
      transform-box: fill-box;
      transform-origin: center;
      will-change: transform, opacity;
    }}
    .morph-shell {{
      d: path("M336 220C470 220 590 230 646 260C690 286 714 330 720 381C724 400 730 416 735 430C742 452 747 470 748 490C752 512 756 535 756 555C758 580 758 605 756 625C754 648 751 670 748 690C744 718 741 742 744 764C749 795 754 822 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z");
    }}
    .soft-shell {{
      transform-origin: 62% 52%;
    }}
    .contact-ball {{
      transform: translate(320px, 0) scaleX(0.96) scaleY(1.04);
    }}
    .cavity-wall {{
      transform: scaleX(0.16) scaleY(0.62);
      opacity: 0;
    }}
    .is-loading .contact-ball {{
      animation: contact-press 2.08s linear 0.08s both;
    }}
    .is-loading .morph-shell {{
      animation: shell-morph 2.08s linear 0.08s both;
    }}
    .is-loading .soft-shell {{
      animation: elastic-recoil 2.08s linear 0.08s both;
    }}
    .is-loading .cavity-wall {{
      animation: cavity-capture 2.08s linear 0.08s both;
    }}
    .is-loading .logo-stage::before {{
      animation: haptic-pulse 2.08s linear 0.08s both;
    }}
    .is-loading .logo-stage::after {{
      animation: settle-glow 2.08s linear 0.08s both;
    }}
    .is-failed .morph-shell {{
      d: path("M336 220C470 220 590 230 646 260C690 286 714 330 709 381C707 400 698 407 681 398C620 378 570 366 530 378C475 394 441 449 433 515C421 612 455 682 519 718C577 748 636 727 688 694C725 671 755 684 770 710C786 740 770 808 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z");
    }}
    .is-failed .contact-ball {{
      transform: none;
    }}
    .is-failed .cavity-wall {{
      transform: none;
      opacity: 1;
    }}
    h1 {{
      margin: 0;
      font-size: 30px;
      line-height: 1;
      font-weight: 760;
      letter-spacing: 0.08em;
    }}
    .expanded-name {{
      max-width: 590px;
      margin: 0;
      color: #526b7f;
      font-size: 14px;
      line-height: 1.45;
      text-wrap: balance;
    }}
    .startup-status {{
      display: grid;
      gap: 2px;
      margin-top: 8px;
    }}
    .status-title {{
      color: #18384d;
      font-size: 14px;
      font-weight: 680;
    }}
    .status-detail {{
      color: #7890a2;
      font-size: 12px;
    }}
    .is-failed .status-title {{ color: #9d433f; }}
    .progress {{
      position: relative;
      width: 184px;
      height: 3px;
      margin-top: 8px;
      overflow: hidden;
      border-radius: 2px;
      background: #dbe8ef;
    }}
    .progress span {{
      position: absolute;
      inset: 0 auto 0 -45%;
      width: 45%;
      border-radius: inherit;
      background: linear-gradient(90deg, #6eddef, #1598c2);
      animation: progress-sweep 1.35s cubic-bezier(0.45, 0, 0.25, 1) infinite;
    }}
    .close-action {{
      margin-top: 8px; min-width: 92px; height: 38px; border-radius: 6px;
      border: 1px solid #a9c4d6; background: #fff; color: #15364c;
      font: inherit; font-weight: 600; cursor: pointer;
    }}
    @keyframes contact-press {{
      0%, 8% {{
        transform: translate(320px, 0) scaleX(0.96) scaleY(1.04);
        animation-timing-function: cubic-bezier(0.38, 0, 0.18, 1);
      }}
      14% {{
        transform: translate(350px, -6px) scaleX(0.94) scaleY(1.06);
        animation-timing-function: cubic-bezier(0.18, 0.82, 0.16, 1);
      }}
      31% {{
        transform: translate(82px, -3px) scaleX(1.045) scaleY(0.97);
        animation-timing-function: cubic-bezier(0.12, 0.76, 0.15, 1);
      }}
      40% {{
        transform: translate(-22px, 2px) scaleX(0.84) scaleY(1.12);
        animation-timing-function: cubic-bezier(0.16, 0.86, 0.20, 1);
      }}
      53% {{
        transform: translate(20px, -5px) scaleX(1.035) scaleY(0.98);
        animation-timing-function: cubic-bezier(0.20, 0.78, 0.22, 1);
      }}
      66% {{
        transform: translate(-10px, 2px) scaleX(0.95) scaleY(1.04);
        animation-timing-function: cubic-bezier(0.22, 0.72, 0.28, 1);
      }}
      78% {{
        transform: translate(6px, -1px) scaleX(1.012) scaleY(0.994);
        animation-timing-function: cubic-bezier(0.24, 0.68, 0.30, 1);
      }}
      89% {{
        transform: translate(-3px, 1px) scaleX(0.987) scaleY(1.014);
        animation-timing-function: cubic-bezier(0.26, 0.62, 0.34, 1);
      }}
      100% {{ transform: translate(0, 0) scaleX(1) scaleY(1); }}
    }}
    @keyframes shell-morph {{
      0%, 16% {{
        d: path("M336 220C470 220 590 230 646 260C690 286 714 330 720 381C724 400 730 416 735 430C742 452 747 470 748 490C752 512 756 535 756 555C758 580 758 605 756 625C754 648 751 670 748 690C744 718 741 742 744 764C749 795 754 822 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z");
      }}
      31% {{
        d: path("M336 220C470 220 590 230 646 260C690 286 714 330 712 381C711 401 705 411 690 407C650 390 610 385 580 396C535 415 512 466 508 521C500 610 540 674 600 704C646 725 686 711 720 692C750 674 770 690 776 712C787 749 771 812 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z");
      }}
      40% {{
        d: path("M336 220C470 220 590 230 646 260C685 284 706 329 700 382C697 409 684 416 660 402C586 374 527 367 483 390C424 420 395 481 392 541C388 640 432 715 502 743C570 770 645 738 704 692C748 661 780 680 788 714C798 750 776 821 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z");
      }}
      53% {{
        d: path("M336 220C470 220 590 230 646 260C691 286 716 330 711 381C709 399 701 407 686 399C633 381 585 370 546 380C493 394 457 444 448 509C435 602 468 669 527 705C581 737 637 721 684 693C719 672 750 684 766 710C783 741 769 808 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z");
      }}
      66% {{
        d: path("M336 220C470 220 590 230 646 260C688 285 711 330 705 382C703 403 693 410 675 400C607 377 556 367 515 381C460 399 426 457 420 523C411 620 449 690 514 724C574 755 638 729 693 693C732 668 761 683 776 711C791 742 773 814 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z");
      }}
      78% {{
        d: path("M336 220C470 220 590 230 646 260C691 286 715 330 710 381C708 400 699 407 682 398C623 379 573 368 533 378C478 393 444 447 436 513C423 609 457 679 520 716C578 746 636 726 687 694C724 671 754 684 769 710C785 740 770 809 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z");
      }}
      90%, 100% {{
        d: path("M336 220C470 220 590 230 646 260C690 286 714 330 709 381C707 400 698 407 681 398C620 378 570 366 530 378C475 394 441 449 433 515C421 612 455 682 519 718C577 748 636 727 688 694C725 671 755 684 770 710C786 740 770 808 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z");
      }}
    }}
    @keyframes elastic-recoil {{
      0%, 31% {{ transform: translateX(0) scaleX(1) scaleY(1); }}
      40% {{ transform: translateX(-8px) scaleX(0.982) scaleY(1.014); }}
      53% {{ transform: translateX(5px) scaleX(1.007) scaleY(0.995); }}
      66% {{ transform: translateX(-3px) scaleX(0.996) scaleY(1.004); }}
      78% {{ transform: translateX(1px) scaleX(1.002) scaleY(0.999); }}
      90%, 100% {{ transform: translateX(0) scaleX(1) scaleY(1); }}
    }}
    @keyframes cavity-capture {{
      0%, 24% {{ transform: scaleX(0.16) scaleY(0.62); opacity: 0; }}
      31% {{ transform: scaleX(0.52) scaleY(0.82); opacity: 0.34; }}
      40% {{ transform: scaleX(1.13) scaleY(1.08); opacity: 1; }}
      53% {{ transform: scaleX(0.91) scaleY(0.97); opacity: 0.84; }}
      66% {{ transform: scaleX(1.045) scaleY(1.02); opacity: 1; }}
      78% {{ transform: scaleX(0.98) scaleY(0.995); opacity: 0.96; }}
      90%, 100% {{ transform: scaleX(1) scaleY(1); opacity: 1; }}
    }}
    @keyframes haptic-pulse {{
      0%, 32% {{ transform: scale(0.82); opacity: 0; }}
      38% {{ transform: scale(0.91); opacity: 0.42; }}
      56% {{ transform: scale(1.23); opacity: 0; }}
      100% {{ transform: scale(1.23); opacity: 0; }}
    }}
    @keyframes settle-glow {{
      0%, 28% {{ transform: scale(0.84); opacity: 0; }}
      38% {{ transform: scale(1.01); opacity: 0.68; }}
      54% {{ transform: scale(1.09); opacity: 0.26; }}
      72%, 100% {{ transform: scale(1.14); opacity: 0; }}
    }}
    @keyframes progress-sweep {{
      0% {{ transform: translateX(0); }}
      100% {{ transform: translateX(325%); }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      .morph-shell, .soft-shell, .contact-ball, .cavity-wall,
      .logo-stage::before, .logo-stage::after, .progress span {{
        animation: none !important;
      }}
      .morph-shell {{
        d: path("M336 220C470 220 590 230 646 260C690 286 714 330 709 381C707 400 698 407 681 398C620 378 570 366 530 378C475 394 441 449 433 515C421 612 455 682 519 718C577 748 636 727 688 694C725 671 755 684 770 710C786 740 770 808 741 868C711 920 650 946 552 957C430 970 294 960 214 942C121 922 89 862 77 776C64 690 73 507 83 419C93 331 129 272 194 242C231 226 278 219 336 220Z") !important;
      }}
      .contact-ball {{ transform: none !important; }}
      .cavity-wall {{ transform: none !important; opacity: 1 !important; }}
      .logo-stage::before, .logo-stage::after {{ display: none; }}
      .progress span {{ left: 0; width: 100%; opacity: 0.72; }}
    }}
  </style>
</head>
<body class="{state_class}">
  <header class="titlebar pywebview-drag-region">
    <img src="{escape(logo_uri)}" alt="">
    <strong>{escape(APP_TITLE)}</strong>
  </header>
  <main>
    <section class="startup" aria-live="polite">
      <div class="logo-stage" aria-hidden="true">
        {_ANIMATED_STARTUP_LOGO}
      </div>
      <h1>{escape(APP_TITLE)}</h1>
      <p class="expanded-name">{escape(APP_EXPANDED_TITLE)}</p>
      <div class="startup-status">
        <span class="status-title">{escape(title)}</span>
        <span class="status-detail">{escape(detail)}</span>
      </div>
      {activity}
      {close_action}
    </section>
  </main>
</body>
</html>"""


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def beta_all_data_runtime_requested() -> bool:
    explicit_values = (
        os.environ.get("TOUCH_LATEST_ALL_DATA_MODEL", ""),
        os.environ.get("TOUCH_BETA_ALL_DATA_MODEL", ""),
    )
    if any(
        str(value).strip().lower() in {"1", "true", "yes", "on"}
        for value in explicit_values
    ):
        return True
    if not is_frozen():
        return False
    marker_roots = (
        Path(sys.executable).resolve().parent,
        Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve(),
    )
    marker_locations = [
        root / filename
        for root in marker_roots
        for filename in (LATEST_RUNTIME_FLAG_FILENAME, BETA_RUNTIME_FLAG_FILENAME)
    ]
    return any(marker.is_file() for marker in marker_locations)


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


def enable_borderless_taskbar_toggle(
    window: Any,
    *,
    user32: Any | None = None,
) -> bool:
    """Restore native taskbar minimize/restore behavior for a frameless form."""

    if os.name != "nt" and user32 is None:
        return False
    native = getattr(window, "native", None)
    handle = getattr(native, "Handle", None)
    if handle is None:
        return False
    try:
        hwnd = int(handle.ToInt64()) if hasattr(handle, "ToInt64") else int(handle)
        if hwnd <= 0:
            return False
        api = user32 or ctypes.WinDLL("user32", use_last_error=True)
        get_window_long = api.GetWindowLongPtrW
        set_window_long = api.SetWindowLongPtrW
        set_window_pos = api.SetWindowPos
        if user32 is None:
            get_window_long.argtypes = [ctypes.c_void_p, ctypes.c_int]
            get_window_long.restype = ctypes.c_ssize_t
            set_window_long.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_ssize_t,
            ]
            set_window_long.restype = ctypes.c_ssize_t
            set_window_pos.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint,
            ]
            set_window_pos.restype = ctypes.c_bool

        current_style = int(get_window_long(hwnd, GWL_STYLE))
        required_style = current_style | WS_SYSMENU | WS_MINIMIZEBOX
        if required_style != current_style:
            set_window_long(hwnd, GWL_STYLE, required_style)
            set_window_pos(
                hwnd,
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
        write_log(
            "Desktop native taskbar toggle enabled "
            f"hwnd={hwnd} style=0x{required_style & 0xFFFFFFFF:08X}"
        )
        return True
    except Exception as exc:
        write_log(
            "Desktop native taskbar toggle setup failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def show_error(title: str, message: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, title, 0x00000010)
    except Exception:
        write_log(f"{title}: {message}")


def configure_runtime_paths() -> Path:
    app_root = bundle_root()
    if beta_all_data_runtime_requested():
        os.environ["TOUCH_LATEST_ALL_DATA_MODEL"] = "true"
        # Legacy environment name remains available to the current backend
        # contract and previously built Beta packages.
        os.environ["TOUCH_BETA_ALL_DATA_MODEL"] = "true"
    os.environ["BAYSPEC_WAVELENGTH_APP_ROOT"] = str(app_root)
    if is_frozen():
        # Keep experiment evidence outside the replaceable application bundle.
        documents_root = Path(
            os.environ.get("TOUCH_DOCUMENTS_ROOT")
            or (Path.home() / "Documents")
        ).expanduser()
        capture_root = documents_root / "TOUCH" / "captures"
        os.environ.setdefault("TOUCH_CAPTURE_OUTPUT_ROOT", str(capture_root))
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
        try:
            # A bound process can reject connections while still owning the
            # port. Only a real bind attempt answers whether a new backend can
            # safely listen here.
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
        return True


def require_fixed_port(port: int) -> None:
    if port_is_free(port):
        return
    message = (
        f"Port {port} is already in use.\n\n"
        "Another TOUCH backend is already running. "
        "Close the old trained-spectrum app instance or stop its Python/uvicorn process, "
        "then start this app again. The wavelength-shift and optical-intensity editions use different ports."
    )
    write_log(message)
    show_error(APP_TITLE, message)
    raise RuntimeError(message)


def select_backend_port(
    preferred_port: int = DEFAULT_PORT,
    *,
    candidate_count: int = FALLBACK_PORT_COUNT,
) -> tuple[int, bool]:
    """Return an available or compatible local port.

    The preferred port can be stranded by a native driver after a USB reset.
    A fallback keeps the desktop app usable while still reusing a healthy TOUCH
    backend when one already owns a candidate port.
    """

    candidate_count = max(1, int(candidate_count))
    for offset in range(candidate_count):
        port = int(preferred_port) + offset
        if port_is_free(port):
            return port, True
        health_url = f"http://127.0.0.1:{port}/api/health"
        if backend_is_ready(health_url, timeout_s=0.55):
            return port, False
        write_log(
            f"Backend port {port} is occupied by an unresponsive or incompatible process"
        )
    message = (
        f"No usable TOUCH backend port was found in "
        f"{preferred_port}-{preferred_port + candidate_count - 1}."
    )
    write_log(message)
    show_error(APP_TITLE, message)
    raise RuntimeError(message)


def health_payload_is_expected(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    common_contract = bool(
        payload.get("ok") is True
        and payload.get("app") == EXPECTED_BACKEND_APP
        and payload.get("mode") == EXPECTED_BACKEND_MODE
        and payload.get("backend_contract_version")
        == EXPECTED_BACKEND_CONTRACT_VERSION
    )
    if not common_contract:
        return False
    if beta_all_data_runtime_requested():
        beta_model = payload.get("all_source_beta_model", {})
        return bool(
            payload.get("all_source_beta_primary") is True
            and payload.get("default_operator_recognition")
            == EXPECTED_BETA_OPERATOR_RECOGNITION
            and payload.get("dynamic_temporal_validation_primary") is False
            and payload.get("static_spectral_fallback_available") is False
            and payload.get("old_model_fallback_enabled") is False
            and isinstance(beta_model, dict)
            and beta_model.get("loaded") is True
            and beta_model.get("old_model_fallback_enabled") is False
        )
    return bool(
        payload.get("dynamic_temporal_validation_primary") is True
        and payload.get("default_operator_recognition")
        == EXPECTED_OPERATOR_RECOGNITION
    )


def run_self_test() -> int:
    """Validate frozen resources without opening hardware, a port, or a window."""

    os.environ["TOUCH_PX6D_AUTO_START"] = "false"
    app_root = configure_runtime_paths()
    config_root = app_root / "config" if is_frozen() else app_root.parent / "config"
    checks: dict[str, object] = {
        "app_root": str(app_root),
        "frozen": is_frozen(),
        "beta_all_data_runtime": beta_all_data_runtime_requested(),
        "frontend_index": (app_root / "frontend" / "index.html").is_file(),
        "frontend_javascript": (app_root / "frontend" / "app.js").is_file(),
        "sdk_helper": (app_root / "sdk_probe" / "BaySpecSdkStream.exe").is_file(),
        "mfbg_intensity_config": (
            config_root / "mfbg_intensity_3x3.yaml"
        ).is_file(),
    }
    try:
        from backend.main import health

        payload = health()
        checks["backend_contract"] = health_payload_is_expected(payload)
        if beta_all_data_runtime_requested():
            beta_model = payload.get("all_source_beta_model", {})
            checks["beta_latest_model_loaded"] = bool(beta_model.get("loaded"))
            checks["old_model_fallback_disabled"] = bool(
                payload.get("old_model_fallback_enabled") is False
                and payload.get("static_spectral_fallback_available") is False
                and payload.get("dynamic_temporal_validation_primary") is False
            )
        else:
            checks["static_model_loaded"] = bool(
                payload.get("trained_static_spectral_model", {}).get("loaded")
            )
            checks["dynamic_model_loaded"] = bool(
                payload.get("dynamic_temporal_shadow", {}).get("loaded")
            )
    except Exception as exc:
        checks["backend_import_error"] = f"{type(exc).__name__}: {exc}"
    ignored_check_keys = {"app_root", "frozen"}
    if not beta_all_data_runtime_requested():
        ignored_check_keys.add("beta_all_data_runtime")
    ok = all(
        value is True
        for key, value in checks.items()
        if key not in ignored_check_keys
    )
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


def wait_until_ready(
    url: str,
    timeout_s: float = 20.0,
    *,
    backend_thread: threading.Thread | None = None,
    server_holder: dict[str, Any] | None = None,
) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            if _read_expected_health(url, timeout_s=1.0):
                return
        except Exception as exc:
            last_error = exc
        if backend_thread is not None and not backend_thread.is_alive():
            startup_error = (server_holder or {}).get("startup_error")
            detail = startup_error or last_error or "backend thread exited"
            raise RuntimeError(f"Backend exited before becoming ready: {detail}")
        time.sleep(0.15)
    raise RuntimeError(f"Backend did not become ready: {last_error}")


def load_uvicorn_module() -> Any:
    """Import Uvicorn inside the backend worker instead of blocking first paint."""

    import uvicorn

    return uvicorn


def run_backend(port: int, server_holder: dict[str, Any]) -> None:
    try:
        uvicorn = load_uvicorn_module()
        from backend.main import app

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        server_holder["server"] = server
        server.run()
    except (Exception, SystemExit) as exc:
        error = f"{type(exc).__name__}: {exc}"
        server_holder["startup_error"] = error
        server_holder["startup_traceback"] = traceback.format_exc()
        write_log(server_holder["startup_traceback"])


def load_application_when_ready(
    window: Any,
    *,
    app_root: Path,
    app_url: str,
    health_url: str,
    backend_thread: threading.Thread | None,
    server_holder: dict[str, Any],
    ownership: dict[str, bool],
    started_at: float,
    backend_port: int = DEFAULT_PORT,
) -> None:
    """Navigate the already-visible startup window once the backend is ready."""

    try:
        wait_until_ready(
            health_url,
            backend_thread=backend_thread,
            server_holder=server_holder,
        )
        if backend_thread is not None and not backend_thread.is_alive():
            ownership["owns_backend"] = False
            write_log(
                "Backend start lost the port race; reusing expected backend "
                f"on port {backend_port}"
            )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        write_log(f"STARTUP_TIMING stage=backend_ready elapsed_ms={elapsed_ms:.1f}")
        window.load_url(app_url)
        write_log(
            "STARTUP_TIMING stage=application_navigation "
            f"elapsed_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
        )
    except Exception:
        write_log(traceback.format_exc())
        window.load_html(startup_document(app_root, failed=True))


def request_backend_shutdown(server_holder: dict[str, Any]) -> bool:
    """Signal the owned Uvicorn server without blocking a webview callback."""

    server = server_holder.get("server")
    if server is None:
        return False
    server.should_exit = True
    return True


def stop_owned_backend(
    server_holder: dict[str, Any],
    backend_thread: threading.Thread | None,
    *,
    graceful_timeout_s: float = 8.0,
    force_timeout_s: float = 2.0,
) -> bool:
    """Wait for lifespan cleanup before the desktop process is allowed to exit."""

    if backend_thread is None:
        return True
    request_backend_shutdown(server_holder)
    if backend_thread is threading.current_thread():
        return False
    backend_thread.join(timeout=max(0.0, float(graceful_timeout_s)))
    if not backend_thread.is_alive():
        return True
    server = server_holder.get("server")
    if server is not None:
        server.should_exit = True
        server.force_exit = True
    backend_thread.join(timeout=max(0.0, float(force_timeout_s)))
    return not backend_thread.is_alive()


class DesktopApi:
    """Small native-only API exposed to the bundled pywebview frontend."""

    def __init__(self, *, initially_maximized: bool = False) -> None:
        self._is_maximized = bool(initially_maximized)
        self._window_zoom = MacStyleTaskbarZoom(
            app_title=APP_TITLE,
            logger=write_log,
        )

    def attach_window(self, window: Any) -> None:
        enable_borderless_taskbar_toggle(window)
        self._window_zoom.attach(window)

    def note_window_maximized(self) -> None:
        self._is_maximized = True

    def note_window_restored(self) -> None:
        self._is_maximized = False
        self._window_zoom.note_restored()

    @staticmethod
    def _desktop_window() -> Any | None:
        return webview.windows[0] if webview.windows else None

    def minimize_window(self) -> dict[str, object]:
        window = self._desktop_window()
        if window is None:
            return {"ok": False, "status": "desktop_window_not_ready"}
        write_log("Desktop command: minimize")
        if not self._window_zoom.minimize(window):
            window.minimize()
        return {"ok": True, "status": "window_minimized"}

    def toggle_maximize_window(self) -> dict[str, object]:
        window = self._desktop_window()
        if window is None:
            return {"ok": False, "status": "desktop_window_not_ready"}
        target_maximized = not self._is_maximized
        write_log(
            "Desktop command: toggle maximize "
            f"current={self._is_maximized} target={target_maximized}"
        )
        # WinForms can leave a frameless window at its restored bounds when
        # using the regular maximize operation. The fullscreen transition
        # explicitly applies the monitor bounds and restores the previous
        # bounds on the next call.
        window.toggle_fullscreen()
        enable_borderless_taskbar_toggle(window)

        # WinForms may emit maximized/restored synchronously. Assign the
        # intended state instead of inverting again after that callback.
        self._is_maximized = target_maximized
        write_log(f"Desktop command complete: maximized={self._is_maximized}")
        return {
            "ok": True,
            "status": "window_maximized" if self._is_maximized else "window_restored",
            "maximized": self._is_maximized,
        }

    def close_window(self) -> dict[str, object]:
        window = self._desktop_window()
        if window is None:
            return {"ok": False, "status": "desktop_window_not_ready"}
        self._window_zoom.detach()
        window.destroy()
        return {"ok": True, "status": "window_closed"}

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
    started_at = time.perf_counter()
    app_root = configure_runtime_paths()
    write_log(f"Starting {APP_TITLE}; app_root={app_root}")
    server_holder: dict[str, Any] = {}
    port, start_backend_after_window = select_backend_port(DEFAULT_PORT)
    health_url = f"http://127.0.0.1:{port}/api/health"
    app_url = f"http://127.0.0.1:{port}/?desktop=1"
    ownership = {"owns_backend": start_backend_after_window}
    backend_thread: threading.Thread | None = None
    try:
        if start_backend_after_window:
            if port != DEFAULT_PORT:
                write_log(
                    f"Preferred backend port {DEFAULT_PORT} is unavailable; "
                    f"starting isolated backend on fallback port {port}"
                )
        else:
            write_log(f"Reusing existing TOUCH temporal backend on port {port}")

        desktop_api = DesktopApi(initially_maximized=False)
        window = webview.create_window(
            APP_TITLE,
            html=startup_document(app_root),
            width=1180,
            height=760,
            min_size=(1024, 680),
            maximized=False,
            background_color="#f5f9fc",
            js_api=desktop_api,
            frameless=True,
            easy_drag=False,
            shadow=True,
        )

        window.events.maximized += desktop_api.note_window_maximized
        window.events.restored += desktop_api.note_window_restored

        def on_shown(window: Any) -> None:
            desktop_api.attach_window(window)
            write_log(
                "STARTUP_TIMING stage=window_shown "
                f"elapsed_ms={(time.perf_counter() - started_at) * 1000.0:.1f}"
            )

        window.events.shown += on_shown

        def on_closed() -> None:
            desktop_api._window_zoom.detach()
            if ownership["owns_backend"]:
                request_backend_shutdown(server_holder)

        def finish_startup() -> None:
            nonlocal backend_thread
            if start_backend_after_window:
                backend_thread = threading.Thread(
                    target=run_backend,
                    args=(port, server_holder),
                    name="touch-uvicorn-backend",
                    daemon=True,
                )
                backend_thread.start()
            load_application_when_ready(
                window,
                app_root=app_root,
                app_url=app_url,
                health_url=health_url,
                backend_thread=backend_thread,
                server_holder=server_holder,
                ownership=ownership,
                started_at=started_at,
                backend_port=port,
            )

        window.events.closed += on_closed
        webview.start(finish_startup, debug=False)
        return 0
    finally:
        if ownership["owns_backend"] and not stop_owned_backend(
            server_holder,
            backend_thread,
        ):
            write_log(
                "Owned backend did not exit after graceful and forced shutdown waits"
            )


if __name__ == "__main__":
    try:
        multiprocessing.freeze_support()
        if "--self-test" in sys.argv:
            raise SystemExit(run_self_test())
        raise SystemExit(main())
    except Exception:
        write_log(traceback.format_exc())
        raise
