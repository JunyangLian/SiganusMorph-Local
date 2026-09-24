from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any, Iterable

import streamlit as st


OBSIDIAN = "#000000"
INKSTONE = "#181818"
FELT_GRAY = "#6D6D6D"
PAPER = "#FFFFFF"
CANVAS = "#F5F5F3"
HAIRLINE = "#D8D8D4"
SUCCESS = "#1F7A4D"
WARNING = "#A86600"
ERROR = "#B42318"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HERO_ASSET = _PROJECT_ROOT / "assets" / "formal_ui" / "iridescent_hero.webp"


@lru_cache(maxsize=1)
def _hero_data_uri() -> str:
    if not _HERO_ASSET.exists():
        return ""
    encoded = base64.b64encode(_HERO_ASSET.read_bytes()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def apply_formal_theme() -> None:
    """Apply the editorial V1.0 design system to public Streamlit pages."""
    st.markdown(
        f"""
        <style>
        :root {{
            --sm-black: {OBSIDIAN}; --sm-ink: {INKSTONE}; --sm-muted: {FELT_GRAY};
            --sm-paper: {PAPER}; --sm-canvas: {CANVAS}; --sm-line: {HAIRLINE};
            --sm-success: {SUCCESS}; --sm-warning: {WARNING}; --sm-error: {ERROR};
            --sm-ease: cubic-bezier(0.19, 1, 0.22, 1);
        }}
        .stApp {{
            background: var(--sm-canvas); color: var(--sm-black);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
        }}
        .block-container {{ max-width: 1120px; padding-top: 28px; padding-bottom: 64px; }}
        h1, h2, h3, h4, p, label {{ letter-spacing: 0 !important; }}
        h1 {{ color: var(--sm-black); font-size: 40px !important; font-weight: 400 !important; line-height: 1.14 !important; }}
        h2 {{ color: var(--sm-black); font-size: 28px !important; font-weight: 400 !important; line-height: 1.2 !important; }}
        h3 {{ color: var(--sm-ink); font-size: 18px !important; font-weight: 600 !important; line-height: 1.3 !important; }}

        [data-testid="stSidebar"] {{ background: var(--sm-paper); border-right: 1px solid var(--sm-line); }}
        [data-testid="stSidebarContent"] {{ padding-top: 18px; }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ margin-bottom: 0; }}
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
            border-radius: 0 !important; min-height: 44px; margin: 0; padding: 10px 16px;
            border-top: 1px solid transparent; border-bottom: 1px solid transparent; color: var(--sm-ink);
            transition: background-color .8s var(--sm-ease), color .8s var(--sm-ease);
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{ background: #F1F1EE; color: var(--sm-black); }}
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {{ background: var(--sm-black); color: var(--sm-paper); }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid var(--sm-line) !important; border-radius: 0 !important;
            background: var(--sm-paper) !important; box-shadow: none !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{ padding: 26px !important; }}

        .sm-hero {{
            background: var(--sm-paper); border-top: 1px solid var(--sm-black); border-bottom: 1px solid var(--sm-line);
            padding: 38px 0 34px; margin-bottom: 28px;
        }}
        .sm-hero-media {{
            position: relative; min-height: 360px; display: flex; flex-direction: column; justify-content: flex-end;
            overflow: hidden; padding: 42px; border: 0; background-color: #101514; background-size: cover; background-position: center;
        }}
        .sm-hero-media::after {{
            content: ""; position: absolute; inset: 0;
            background: linear-gradient(90deg, rgba(0,0,0,.76) 0%, rgba(0,0,0,.33) 54%, rgba(0,0,0,.08) 100%);
            pointer-events: none;
        }}
        .sm-hero-media > * {{ position: relative; z-index: 1; }}
        .sm-hero-media .sm-eyebrow, .sm-hero-media .sm-hero-title {{ color: var(--sm-paper); }}
        .sm-hero-media .sm-hero-subtitle {{ color: rgba(255,255,255,.78); }}
        .sm-eyebrow {{ color: var(--sm-muted); font-size: 11px; font-weight: 600; text-transform: uppercase; margin-bottom: 14px; }}
        .sm-hero-title {{
            max-width: 850px; color: var(--sm-black); font-size: clamp(34px, 5vw, 58px);
            font-weight: 300; line-height: 1.08; margin: 0;
        }}
        .sm-hero-subtitle {{ color: var(--sm-muted); font-size: 16px; line-height: 1.6; max-width: 760px; margin-top: 18px; }}

        .sm-section-title {{
            display: flex; gap: 14px; align-items: flex-start; padding-bottom: 14px;
            margin-bottom: 18px; border-bottom: 1px solid var(--sm-line);
        }}
        .sm-step {{
            min-width: 34px; height: 34px; border-radius: 75px; border: 1px solid var(--sm-black);
            background: transparent; color: var(--sm-black); display: inline-flex; justify-content: center;
            align-items: center; font-weight: 600; font-size: 12px;
        }}
        .sm-section-heading {{ color: var(--sm-black); font-size: 18px; font-weight: 600; margin: 0; }}
        .sm-section-subtitle {{ color: var(--sm-muted); font-size: 13px; margin-top: 3px; line-height: 1.45; }}

        .sm-card, .sm-feature-card {{
            background: var(--sm-paper); border: 1px solid var(--sm-line); border-radius: 0;
            padding: 24px; box-shadow: none;
        }}
        .sm-feature-card {{ min-height: 156px; }}
        .sm-feature-title {{ color: var(--sm-black); font-size: 16px; font-weight: 600; margin-bottom: 12px; }}
        .sm-feature-body {{ color: var(--sm-muted); font-size: 14px; line-height: 1.6; }}
        .sm-feature-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
        .sm-flow-grid {{
            display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
            border-top: 1px solid var(--sm-black); border-bottom: 1px solid var(--sm-black);
        }}
        .sm-flow-card {{ min-height: 148px; padding: 22px 18px; border-right: 1px solid var(--sm-line); background: var(--sm-paper); }}
        .sm-flow-card:last-child {{ border-right: 0; }}
        .sm-flow-number {{ color: var(--sm-muted); font-size: 11px; margin-bottom: 30px; }}
        .sm-flow-title {{ color: var(--sm-black); font-size: 15px; font-weight: 600; margin-bottom: 8px; }}
        .sm-flow-body {{ color: var(--sm-muted); font-size: 12px; line-height: 1.5; }}

        .sm-badge {{
            display: inline-flex; align-items: center; border-radius: 75px; padding: 5px 11px;
            font-size: 11px; font-weight: 600; border: 1px solid var(--sm-line); white-space: nowrap; background: transparent;
        }}
        .sm-badge-neutral, .sm-badge-info {{ color: var(--sm-ink); border-color: #808080; }}
        .sm-badge-success {{ color: var(--sm-success); border-color: var(--sm-success); }}
        .sm-badge-warning {{ color: var(--sm-warning); border-color: var(--sm-warning); }}
        .sm-badge-error {{ color: var(--sm-error); border-color: var(--sm-error); }}

        .sm-metric-grid {{
            display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0;
            border-top: 1px solid var(--sm-line); border-left: 1px solid var(--sm-line); margin: 18px 0 22px;
        }}
        .sm-metric-card {{
            background: var(--sm-paper); border-right: 1px solid var(--sm-line); border-bottom: 1px solid var(--sm-line);
            border-radius: 0; padding: 16px;
        }}
        .sm-metric-label {{ color: var(--sm-muted); font-size: 11px; margin-bottom: 8px; }}
        .sm-metric-value {{ color: var(--sm-black); font-size: 20px; font-weight: 400; overflow-wrap: anywhere; }}
        .sm-help {{ color: var(--sm-muted); font-size: 13px; line-height: 1.55; padding-top: 11px; }}
        .sm-workspace-title {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 14px; }}

        div.stButton > button, div.stDownloadButton > button, [data-testid="stPageLink-NavLink"] {{
            border-radius: 75px !important; box-shadow: none !important; font-weight: 500 !important; min-height: 42px;
            transition: transform .8s var(--sm-ease), background-color .8s var(--sm-ease), color .8s var(--sm-ease) !important;
        }}
        div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {{
            background: var(--sm-black) !important; border-color: var(--sm-black) !important; color: var(--sm-paper) !important;
        }}
        div.stButton > button[kind="secondary"], div.stDownloadButton > button {{
            background: transparent !important; border-color: var(--sm-black) !important; color: var(--sm-black) !important;
        }}
        div.stButton > button:hover, div.stDownloadButton > button:hover {{
            transform: translateY(-1px); background: var(--sm-black) !important; color: var(--sm-paper) !important;
        }}
        div.stButton > button:disabled {{ border-color: var(--sm-line) !important; color: #9A9A9A !important; background: transparent !important; }}

        [data-testid="stFileUploader"] {{ background: var(--sm-paper); border: 1px dashed #808080; border-radius: 0; padding: 14px; }}
        [data-testid="stFileUploaderDropzone"] {{ background: transparent !important; border-radius: 0 !important; }}
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stTextArea"] textarea {{
            border-radius: 0 !important; border-color: var(--sm-line) !important;
            background: var(--sm-paper) !important; box-shadow: none !important;
        }}
        [data-testid="stDataFrame"] {{ border: 1px solid var(--sm-line); border-radius: 0; overflow: hidden; }}
        [data-testid="stExpander"] {{ border-radius: 0 !important; border-color: var(--sm-line) !important; background: var(--sm-paper); }}
        [data-testid="stAlert"] {{ border-radius: 0 !important; box-shadow: none !important; }}
        .sm-link-card a {{ text-decoration: none; color: var(--sm-black); font-weight: 600; }}

        @media (max-width: 760px) {{
            .block-container {{ padding-top: 18px; padding-left: 16px; padding-right: 16px; }}
            .sm-hero-media {{ min-height: 300px; padding: 26px; }}
            .sm-hero-title {{ font-size: 34px; }}
            .sm-metric-grid {{ grid-template-columns: 1fr; }}
            .sm-feature-grid, .sm-flow-grid {{ grid-template-columns: 1fr; }}
            .sm-flow-card {{ min-height: auto; border-right: 0; border-bottom: 1px solid var(--sm-line); }}
            div[data-testid="stVerticalBlockBorderWrapper"] > div {{ padding: 18px !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div style="padding:8px 16px 24px;border-bottom:1px solid #D8D8D4;margin-bottom:8px;">
          <div style="font-weight:600;color:#000;font-size:15px;">SiganusMorph V1.0</div>
          <div style="color:#6D6D6D;font-size:11px;margin-top:5px;">蓝子鱼形态测量系统</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, eyebrow: str = "", *, media: bool = False) -> None:
    eyebrow_html = f'<div class="sm-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    media_class = " sm-hero-media" if media else ""
    background = ""
    if media and (data_uri := _hero_data_uri()):
        background = f' style="background-image:url(\'{data_uri}\')"'
    st.markdown(
        f"""
        <section class="sm-hero{media_class}"{background}>
            {eyebrow_html}
            <h1 class="sm-hero-title">{escape(title)}</h1>
            <div class="sm-hero-subtitle">{escape(subtitle)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def step_header(step: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="sm-section-title">
            <span class="sm-step">{escape(step)}</span>
            <div><div class="sm-section-heading">{escape(title)}</div><div class="sm-section-subtitle">{escape(subtitle)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(title: str, body: str) -> str:
    return '<div class="sm-feature-card">' + f'<div class="sm-feature-title">{escape(title)}</div>' + f'<div class="sm-feature-body">{escape(body)}</div></div>'


def status_badge(label: str, status: str = "neutral") -> str:
    status_class = {"success": "success", "warning": "warning", "error": "error", "info": "info"}.get(status, "neutral")
    return f'<span class="sm-badge sm-badge-{status_class}">{escape(label)}</span>'


def metric_card(label: str, value: Any, suffix: str = "") -> str:
    display = "—" if value in ("", None) else str(value)
    return '<div class="sm-metric-card">' + f'<div class="sm-metric-label">{escape(label)}</div>' + f'<div class="sm-metric-value">{escape(display)}{escape(suffix)}</div></div>'


def metric_grid(items: Iterable[tuple[str, Any, str]]) -> None:
    html = "".join(metric_card(label, value, suffix) for label, value, suffix in items)
    st.markdown(f'<div class="sm-metric-grid">{html}</div>', unsafe_allow_html=True)


def fmt_mm(value: Any, digits: int = 2) -> str:
    try:
        if value in ("", None):
            return "—"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"
