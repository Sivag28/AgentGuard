"""Streamlit interface for the local AgentGuard investigation harness."""

import json
import html
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

from agent import MODEL, run_agent
from tools import MAX_FILE_BYTES, get_log_context, get_test_summary, validate_log_text


REPORT_HISTORY_PATH = Path(__file__).resolve().parent / ".agentguard" / "history.json"


def load_saved_reports() -> list[dict]:
    """Load locally saved report records; uploaded source logs are never stored here."""
    try:
        payload = json.loads(REPORT_HISTORY_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def save_report(record: dict) -> None:
    reports = load_saved_reports()
    reports = [item for item in reports if item.get("id") != record.get("id")]
    reports.insert(0, record)
    REPORT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = REPORT_HISTORY_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(reports[:100], ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(REPORT_HISTORY_PATH)


def log_timeline_events(log_text: str) -> list[tuple[int, str, str]]:
    """Return exact source lines for failure, retry, recovery, and pipeline events."""
    event_pattern = re.compile(
        r"\b(?:FAIL|ERROR|FATAL|WARN|PASS)\b|\bRetry\s+(?:attempt|\d+)|\bPIPELINE\s+BLOCKED\b",
        re.IGNORECASE,
    )
    events = []
    for line_number, line in enumerate(log_text.splitlines(), start=1):
        if event_pattern.search(line):
            kind_match = re.search(r"\b(FAIL|ERROR|FATAL|WARN|PASS)\b|\b(PIPELINE\s+BLOCKED)\b|\b(Retry)\b", line, re.IGNORECASE)
            kind = next((value for value in kind_match.groups() if value), "EVENT").upper() if kind_match else "EVENT"
            events.append((line_number, kind, line.strip()))
    return events


st.set_page_config(
    page_title="AgentGuard | CI Investigator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ag-ink:#15243b; --ag-muted:#63738a; --ag-blue:#2563eb; }
      .stApp { background:#f4f7fb; color:var(--ag-ink); }
      [data-testid="stHeader"] { background:rgba(244,247,251,.9); }
      .block-container { max-width:1320px; padding:2.2rem 2.5rem 4rem; }
      [data-testid="stSidebar"] { background:#101b2d; border-right:1px solid #24324a; }
      [data-testid="stSidebar"] * { color:#e5edf8; }
      [data-testid="stSidebar"] [data-testid="stAlert"] * { color:#17233b; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { letter-spacing:-.04em; }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#9aabc2; }
      h1, h2, h3 { letter-spacing:-.035em; color:#14233a; }
      h2 { font-size:1.5rem !important; }
      .ag-hero { position:relative; overflow:hidden; padding:2.4rem 2.6rem;
        border:1px solid #263e65; border-radius:24px;
        background:radial-gradient(ellipse at 88% 0%,#3567c8 0%,transparent 38%),
          linear-gradient(120deg,#111e33 0%,#17345c 55%,#1d4ed8 120%);
        color:white; margin:0 0 2rem; box-shadow:0 18px 45px #162b4a20; }
      .ag-hero h1 { color:white; font-size:2.55rem; line-height:1.12; margin:0 0 .65rem; }
      .ag-hero p { color:#d3e0f4; max-width:760px; margin:0; font-size:1.04rem; line-height:1.65; }
      .ag-kicker { text-transform:uppercase; letter-spacing:.14em;
        font-size:.69rem; font-weight:750; color:#a9c8ff; margin-bottom:.8rem; }
      div[data-testid="stMetric"] { min-height:112px; background:#fff; border:1px solid #e3e9f2;
        border-radius:16px; padding:1.05rem 1.2rem; box-shadow:0 5px 18px #162b4a08; }
      div[data-testid="stMetricLabel"] p { color:#65758c; font-size:.78rem; font-weight:650; }
      div[data-testid="stMetricValue"] { color:#172942; font-weight:720; }
      .ag-section { display:flex; align-items:center; gap:.7rem; margin:.3rem 0 .2rem; }
      .ag-step { display:inline-flex; width:1.7rem; height:1.7rem; align-items:center; justify-content:center;
        border-radius:50%; background:#e6efff; color:#2456b8; font-size:.78rem; font-weight:750; }
      .ag-section-title { color:#172942; font-size:1.18rem; font-weight:730; letter-spacing:-.025em; }
      .ag-section-note { margin:.1rem 0 .9rem 2.4rem; color:#6c7c91; font-size:.9rem; }
      .ag-report { background:linear-gradient(180deg,#fff 0%,#fbfcfe 100%); border:1px solid #dfe7f1;
        border-radius:16px; padding:1.25rem 1.5rem; box-shadow:0 8px 24px #162b4a08; }
      .ag-report p { line-height:1.75; }
      .ag-label { color:#69809f; font-size:.68rem; letter-spacing:.11em; text-transform:uppercase; font-weight:750; }
      .ag-overline { margin:-.45rem 0 1.2rem; color:#718198; font-size:.88rem; }
      .ag-report-meta { display:flex; flex-wrap:wrap; gap:.55rem; margin:.15rem 0 .9rem; }
      .ag-chip { display:inline-flex; align-items:center; border:1px solid #dce5f0; border-radius:999px;
        background:#fff; padding:.34rem .68rem; color:#566981; font-size:.78rem; font-weight:620; }
      [data-testid="stChatMessage"] { border:1px solid #e3e9f2; border-radius:16px; padding:.9rem 1.1rem; }
      [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { line-height:1.72; }
      [data-testid="stChatInput"] textarea { border-radius:15px !important; }
      .ag-welcome { max-width:780px; margin:2rem auto 1.3rem; text-align:center; }
      .ag-welcome h2 { font-size:1.8rem !important; margin-bottom:.4rem; }
      .ag-welcome p { color:#708096; font-size:.96rem; }
      [data-testid="stStatusWidget"] { border-radius:14px; }
      [data-testid="stDataFrame"] { border:1px solid #e1e8f1; border-radius:12px; overflow:hidden; }
      [data-testid="stRadio"] > label, [data-testid="stSelectbox"] > label,
      [data-testid="stFileUploader"] > label { font-weight:650; color:#263a56; }
      [data-testid="stRadio"] [role="radiogroup"] { gap:.6rem; }
      [data-testid="stRadio"] [role="radiogroup"] label { background:white; border:1px solid #e1e8f1;
        border-radius:12px; padding:.55rem .9rem; }
      [data-testid="stSelectbox"] [data-baseweb="select"] > div,
      [data-testid="stTextArea"] textarea { border-radius:12px; border-color:#d8e1ed; }
      [data-testid="stTextArea"] textarea { background:#fff; line-height:1.6; }
      [data-testid="stFileUploader"] section { border:1px dashed #b9c8dc; border-radius:14px; background:#fff; }
      [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] { background:#111d30 !important;
        border:1px solid #2b4865 !important; border-radius:10px; }
      [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
      [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] p { color:#dcecff !important; }
      .stButton > button { border-radius:11px; font-weight:650; transition:all .15s ease; }
      .stButton > button[kind="primary"] { padding:.72rem 1.2rem; box-shadow:0 5px 14px #2563eb30; }
      .stButton > button[kind="primary"]:hover { box-shadow:0 8px 20px #2563eb40; transform:translateY(-1px); }
      [data-testid="stAlert"] { border-radius:12px; }
      [data-testid="stExpander"] { background:#fff; border:1px solid #e3e9f2; border-radius:13px; }
      [data-testid="stCode"] { border:1px solid #e3e9f2; border-radius:12px; }
      hr { border-color:#e1e8f1; margin:1.7rem 0; }
      @media (max-width: 800px) {
        .block-container { padding:1.3rem 1rem 2.5rem; }
        .ag-hero { padding:1.7rem 1.4rem; border-radius:18px; }
        .ag-hero h1 { font-size:2rem; }
      }
      /* Neon agent workspace */
      .stApp { background:radial-gradient(ellipse at 52% -12%,#13284a 0%,#0a1020 48%,#080d18 100%); color:#e7efff; }
      [data-testid="stHeader"] { background:rgba(8,13,24,.78); }
      [data-testid="stAppViewContainer"] { background:transparent; }
      .block-container { max-width:1240px; padding-top:1.6rem; }
      [data-testid="stMarkdownContainer"], [data-testid="stText"] { color:#dce6f8; }
      h1, h2, h3, h4 { color:#f4f7ff; }
      h2 { font-size:1.35rem !important; }
      [data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1423 0%,#0a101c 100%); border-right:1px solid #1b2b45; }
      [data-testid="stSidebar"] * { color:#dce7fa; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { color:#f0f8ff; text-shadow:0 0 18px #21d9ff35; }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#8397b7; }
      [data-testid="stSidebar"] [data-testid="stCode"] { background:#101a2b; border-color:#263b5b; }
      [data-testid="stSidebar"] .stButton > button { background:#111f33; border-color:#2a5671; }
      [data-testid="stSidebar"] .stButton > button:hover { background:#123044; }
      .ag-hero { padding:1.8rem 2.2rem; margin-bottom:1.3rem; border:1px solid #294f79;
        background:radial-gradient(ellipse at 82% 8%,#6638bd70 0%,transparent 32%),
          radial-gradient(ellipse at 96% 90%,#00d5ff28 0%,transparent 38%),
          linear-gradient(115deg,#0d1728 0%,#101d35 58%,#142349 100%);
        box-shadow:0 0 0 1px #29dfff12,0 20px 65px #0008,0 0 45px #1775ff12; }
      .ag-hero h1 { color:#f6f8ff; font-size:2.25rem; text-shadow:0 0 24px #58d6ff24; }
      .ag-hero p { color:#aebfda; }
      .ag-kicker { color:#64e8ff; }
      .ag-section-title { color:#eff5ff; }
      .ag-section-note, .ag-overline { color:#8295b3; }
      .ag-step { color:#60efff; background:#0f3345; box-shadow:0 0 14px #00d9ff22; }
      div[data-testid="stMetric"] { background:linear-gradient(145deg,#111d30,#0e1727); border-color:#263852;
        box-shadow:inset 0 1px #ffffff08,0 10px 30px #0003; }
      div[data-testid="stMetricLabel"] p { color:#8ca0be; }
      div[data-testid="stMetricValue"] { color:#e9f4ff; }
      [data-testid="stRadio"] > label, [data-testid="stSelectbox"] > label,
      [data-testid="stFileUploader"] > label { color:#b8c9e3; }
      [data-testid="stRadio"] [role="radiogroup"] label { background:#111c2e; border-color:#293e5c; }
      [data-testid="stRadio"] [role="radiogroup"] label:hover { border-color:#38d8f2; box-shadow:0 0 16px #22d3ee18; }
      [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
        min-height:2.6rem; background:#0d1727; border-color:#1e3048; }
      [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has([aria-checked="true"]) {
        background:linear-gradient(100deg,#103a5d,#142a4a); border-color:#267bb0;
        box-shadow:inset 0 0 20px #00bfff10,0 0 16px #00bfff12; }
      [data-testid="stSelectbox"] [data-baseweb="select"] > div,
      [data-testid="stTextArea"] textarea, [data-testid="stChatInput"] textarea {
        color:#e8f2ff !important; background:#0d1727 !important; border-color:#2b405d !important; }
      [data-baseweb="select"] > div { background:#0d1727 !important; border-color:#2b405d !important; }
      [data-baseweb="select"] *, [data-baseweb="popover"] * { color:#dce9fb !important; }
      [data-baseweb="popover"] ul, [data-baseweb="popover"] [role="listbox"] { background:#101b2c !important; }
      [data-baseweb="popover"] [role="option"]:hover { background:#17344a !important; }
      [data-testid="stTextArea"] textarea::placeholder, [data-testid="stChatInput"] textarea::placeholder { color:#7689a6; }
      [data-testid="stFileUploader"] section { background:#0e192a; border-color:#31516f; }
      [data-testid="stFileUploader"] section { color:#dcecff; }
      [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] { background:#111d30 !important;
        border:1px solid #31516f !important; }
      [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * { color:#e2efff !important; }
      .stButton > button { color:#dceaff; background:#111d30; border:1px solid #2b405c;
        white-space:normal; height:auto; min-height:2.7rem; line-height:1.35; }
      .stButton > button > div, .stButton > button span { white-space:normal !important;
        overflow:visible !important; text-overflow:clip !important; }
      .stButton > button > div { width:100%; }
      .stButton > button p { white-space:normal !important; overflow:visible !important;
        text-overflow:clip !important; display:block; }
      .stButton > button:hover { color:#6becff; border-color:#35d9f1; background:#12253a; }
      .stButton > button[kind="primary"] { color:#06131d; border:0; font-weight:760;
        background:linear-gradient(100deg,#56edff,#8a7dff); box-shadow:0 0 22px #26dfff42; }
      .stButton > button[kind="primary"]:hover { color:#06131d; box-shadow:0 0 30px #26dfff70; }
      [data-testid="stChatMessage"] { background:linear-gradient(135deg,#101b2d,#0e1726); border:1px solid #243955;
        box-shadow:0 8px 28px #0002; max-width:1020px; width:100%; margin-left:auto; margin-right:auto; }
      [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        max-width:76%; margin-left:auto; margin-right:max(calc((100% - 1020px) / 2), 0px);
        border-color:#28677d; background:linear-gradient(120deg,#102b42,#11213a); }
      [data-testid="stChatInput"] > div { background:#0d1727; border:1px solid #31516f; border-radius:17px;
        box-shadow:0 0 0 1px #27dfff12,0 0 26px #174aff13; }
      [data-testid="stChatInput"] > div:focus-within { border-color:#45dff5; box-shadow:0 0 0 1px #45dff544,0 0 32px #174aff22; }
      [data-testid="stBottomBlockContainer"] { background:#080d18 !important; border-top:1px solid #1d3049;
        box-shadow:0 -14px 35px #080d18cc; }
      [data-testid="stBottom"], footer { background-color:#080d18 !important; }
      [data-testid="stBottomBlockContainer"] > div,
      [data-testid="stBottomBlockContainer"] [data-testid="stChatInput"] { background:transparent !important; }
      [data-testid="stBottomBlockContainer"] [data-testid="stChatInput"] { max-width:1080px; margin-left:auto; margin-right:auto; }
      [data-testid="stBottomBlockContainer"] > div > div { background-color:transparent !important; }
      [data-testid="stBottomBlockContainer"]::before { background:linear-gradient(transparent,#080d18) !important; }
      [data-testid="stChatInput"] button { color:#06131d !important; background:#58eaff !important;
        border:0 !important; box-shadow:0 0 18px #43ddf044; }
      [data-testid="stRadio"] [role="radio"][aria-checked="true"] { background:#113448; border-color:#43dff2; }
      [data-testid="stRadio"] input[type="radio"] { accent-color:#54eaff !important; }
      [data-testid="stFileUploader"] button { color:#d9eaff !important; background:#17253a !important; border:1px solid #304866 !important; }
      [data-testid="stAlert"] { background:#111d2e !important; border:1px solid #304663 !important; }
      [data-testid="stAlert"] * { color:#dce9fb !important; }
      .ag-side-note { padding:.8rem .9rem; background:#101c2d; border:1px solid #273b56;
        border-left:3px solid #46dff4; border-radius:10px; color:#a9bbd4; font-size:.84rem; line-height:1.55; }
      .ag-build-note { margin:.1rem 0 1.2rem; color:#8296b4; font-size:.84rem; }
      .ag-context-card { display:flex; flex-wrap:wrap; align-items:center; gap:0;
        margin:.35rem 0 .9rem; border:1px solid #24415d; border-radius:15px; overflow:hidden;
        background:linear-gradient(105deg,#101b2c 0%,#0d1929 66%,#11243a 100%);
        box-shadow:inset 0 1px #ffffff08,0 10px 28px #0003,0 0 24px #00d9ff08; }
      .ag-context-item { flex:1 1 150px; min-width:130px; padding:.75rem 1rem; border-right:1px solid #20344c; }
      .ag-context-item:last-child { border:0; }
      .ag-context-label { display:block; margin-bottom:.22rem; color:#7891b1; font-size:.65rem;
        font-weight:750; letter-spacing:.12em; text-transform:uppercase; }
      .ag-context-value { display:block; overflow:hidden; color:#e8f4ff; font-size:.98rem;
        font-weight:680; text-overflow:ellipsis; white-space:nowrap; }
      .ag-context-status.failed { color:#ff8ea1; }
      .ag-context-status.success { color:#63f2d1; }
      .ag-context-note { flex:1 0 100%; padding:.52rem 1rem; border-top:1px solid #20344c;
        color:#8ba1bd; font-size:.78rem; }
      .ag-source-card { display:flex; align-items:center; gap:.9rem; min-width:0; min-height:94px; padding:1rem 1.1rem;
        border:1px solid #24415d; border-radius:15px; background:linear-gradient(135deg,#102039,#0d1929);
        box-shadow:0 10px 28px #0003; }
      .ag-source-card > div:nth-child(2) { min-width:0; }
      .ag-source-icon { display:flex; flex:0 0 46px; height:46px; align-items:center; justify-content:center;
        border-radius:13px; background:#173553; color:#6fceff; font-size:1.35rem; }
      .ag-source-title { color:#7891b1; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; font-weight:750; }
      .ag-source-name { color:#e8f4ff; font-size:.95rem; font-weight:680; margin:.22rem 0;
        overflow-wrap:anywhere; word-break:break-word; }
      .ag-source-detail { color:#8ba1bd; font-size:.76rem; }
      .ag-source-ready { margin-left:auto; align-self:flex-start; color:#50f0d0; font-size:1.1rem; }
      .ag-prompt-panel { margin:.25rem 0 .6rem; padding:1.15rem 1.15rem .85rem; border:1px solid #203b5a;
        border-radius:17px; background:linear-gradient(145deg,#0d1d33,#0d1727 58%,#111d31);
        box-shadow:inset 0 1px #ffffff08,0 14px 36px #0003,0 0 28px #146bff0d; }
      .ag-prompt-heading { display:flex; gap:.8rem; align-items:flex-start; margin-bottom:.9rem; }
      .ag-prompt-spark { color:#48caff; font-size:1.6rem; line-height:1.1; text-shadow:0 0 18px #38beff88; }
      .ag-prompt-heading h3 { margin:0 0 .2rem; color:#f0f6ff; font-size:1.15rem; }
      .ag-prompt-heading p { margin:0; color:#9bb0cc; font-size:.82rem; line-height:1.5; }
      .ag-prompt-card { min-height:150px; padding:.9rem; border:1px solid #244260; border-radius:13px;
        background:linear-gradient(145deg,#10213a,#0d192a); }
      .ag-prompt-card h4 { margin:.2rem 0 .4rem; color:#e7f2ff; font-size:.9rem; }
      .ag-prompt-card p { min-height:3.4rem; margin:0 0 .5rem; color:#91a7c5; font-size:.78rem; line-height:1.5; }
      [data-testid="stVerticalBlockBorderWrapper"] { border:1px solid #203b5a !important;
        border-radius:14px !important; background:linear-gradient(145deg,#101f34,#0d1727) !important; }
      [data-testid="stVerticalBlockBorderWrapper"] > div { gap:.5rem; }
      [data-testid="stVerticalBlockBorderWrapper"] .ag-prompt-card { min-height:126px; padding:.15rem .1rem .2rem;
        border:0; background:transparent; }
      div[data-testid="stHorizontalBlock"]:has(.stButton) .stButton > button {
        background:linear-gradient(140deg,#10253a,#111d31); border-color:#2b526d;
        box-shadow:inset 0 1px #ffffff08,0 5px 18px #0002; }
      div[data-testid="stHorizontalBlock"]:has(.stButton) .stButton > button:hover {
        background:linear-gradient(140deg,#12344a,#18203a); box-shadow:0 0 20px #20dfff22; }
      [data-testid="stExpander"] { background:#0f1929; border-color:#253953; }
      [data-testid="stCode"] { background:#0a1220; border-color:#243650; }
      [data-testid="stDataFrame"] { border-color:#263b57; }
      [data-testid="stAlert"] { background:#111d2e; border-color:#2b405d; color:#dce7f8; }
      .ag-welcome { margin:2.5rem auto 1rem; }
      .ag-welcome h2 { font-size:1.85rem !important; text-shadow:0 0 24px #54eaff24; }
      .ag-welcome p { color:#91a4c1; }
      .ag-chip { color:#b7cae4; background:#101b2b; border-color:#2a405e; }
      .ag-hero { padding:.72rem 1.25rem; margin-bottom:.5rem; border-radius:15px; }
      .ag-hero h1 { font-size:1.5rem; margin-bottom:.12rem; }
      .ag-hero p { font-size:.8rem; line-height:1.35; }
      .ag-kicker { margin-bottom:.35rem; }
      .ag-welcome { margin:.45rem auto .35rem; }
      .ag-welcome h2 { font-size:1.55rem !important; margin-bottom:.25rem; }
      .ag-welcome p { margin-bottom:.35rem; font-size:.86rem; }
      .ag-context-item { padding:.6rem .8rem; }
      .block-container { padding-top:3.1rem !important; padding-bottom:6rem; }
      [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:.65rem !important; }
      hr { margin:.2rem 0 !important; }
      .ag-welcome { margin:.15rem auto .2rem; }
      [data-testid="stHorizontalBlock"]:has(.stButton) .stButton > button { min-height:3rem; padding:.55rem .7rem; }
      hr { border-color:#1e304a; }

    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    if "agentguard_page_target" in st.session_state:
        st.session_state["agentguard_page"] = st.session_state.pop("agentguard_page_target")
    page = st.session_state.setdefault("agentguard_page", "Investigations")
    st.markdown("## 🛡️ AgentGuard")
    st.caption("AI-assisted CI failure investigation")
    st.divider()
    st.markdown("**Workspace**")
    page = st.radio(
        "Workspace", ["Overview", "Investigations", "Evidence", "Settings"],
        key="agentguard_page", label_visibility="collapsed",
    )
    st.divider()
    if st.button("New conversation", use_container_width=True):
        for key in (
            "agentguard_history", "agentguard_result", "agentguard_result_log",
            "agentguard_result_log_text", "agentguard_result_source_label", "agentguard_task",
            "agentguard_suggested_prompt", "investigation_chat", "agentguard_report_id",
            "agentguard_archived_report", "agentguard_archived_metadata", "selected_evidence_line",
        ):
            st.session_state.pop(key, None)
        st.session_state["agentguard_page_target"] = "Investigations"
        st.rerun()
    st.markdown("**Build source**")
    uploaded_text = None
    upload_error = None
    uploaded_name = None
    uploaded_file = st.file_uploader("Upload a CI log (.txt, maximum 1 MB)", type=["txt"])
    st.caption("Uploaded logs stay in session memory. Investigation reports and answers are saved locally with citations; original log files are not stored. Ollama synthesizes the answer (local by default).")
    if uploaded_file is not None:
        uploaded_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(uploaded_file.name).name)[:120]
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) > MAX_FILE_BYTES:
            upload_error = "The uploaded log exceeds the 1 MB limit."
        else:
            try:
                uploaded_text = validate_log_text(file_bytes.decode("utf-8-sig"))
            except (UnicodeDecodeError, ValueError) as error:
                upload_error = f"The upload must be a non-empty UTF-8 text log: {error}"
    if upload_error:
        st.error(upload_error)
    elif uploaded_file is None:
        st.info("Upload a .txt CI log to start an investigation.")

    if uploaded_text is not None:
        selected_log = "logs.txt"  # Tool API fallback; uploaded content takes precedence in memory.
        selected_source_label = f"uploaded log '{uploaded_name}' (temporary)"
        source_log_text = uploaded_text
        build_id_match = re.search(r"^\s*Build\s+ID\s*:\s*#?([^\s]+)", uploaded_text, re.IGNORECASE | re.MULTILINE)
        project_match = re.search(r"^\s*Project\s*:\s*(.+?)\s*$", uploaded_text, re.IGNORECASE | re.MULTILINE)
        status_match = re.search(
            r"\b(?:RESULT\s+)?Build\s+status\s*:\s*([^\r\n]+)", uploaded_text,
            re.IGNORECASE,
        )
        build_id = build_id_match.group(1) if build_id_match else uploaded_name or "Uploaded log"
        project_name = project_match.group(1) if project_match else "User supplied"
        build_status = status_match.group(1) if status_match else "Not provided"
    else:
        selected_log = "logs.txt"
        selected_source_label = "uploaded log"
        source_log_text = None
        build_id, project_name, build_status = "Uploaded log", "User supplied", "Not provided"

    saved_reports = load_saved_reports()
    if saved_reports:
        st.divider()
        st.markdown("**Saved investigations**")
        report_by_id = {str(item.get("id")): item for item in saved_reports}
        chosen_report_id = st.selectbox(
            "Saved report", list(report_by_id),
            format_func=lambda report_id: (
                f"{report_by_id[report_id].get('created_at', 'Saved report')} · "
                f"{report_by_id[report_id].get('task', 'Investigation')[:42]} · "
                f"{report_by_id[report_id].get('source_name', 'log')}"
            ),
            label_visibility="collapsed", key="saved_report_choice",
        )
        delete_col, _ = st.columns([1, 5])
        with delete_col:
            delete_selected = st.button(
                "🗑️ Delete", help="Delete the selected saved investigation",
                key=f"delete_saved_report_{chosen_report_id}", use_container_width=True,
            )
        if delete_selected:
            remaining_reports = [
                report for report in saved_reports
                if str(report.get("id")) != chosen_report_id
            ]
            temporary_path = REPORT_HISTORY_PATH.with_suffix(".tmp")
            try:
                temporary_path.write_text(
                    json.dumps(remaining_reports, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary_path.replace(REPORT_HISTORY_PATH)
                st.session_state.pop("saved_report_choice", None)
                st.rerun()
            except OSError as error:
                st.error(f"Could not delete the saved investigation: {error}")
        if st.button("Open saved report", use_container_width=True):
            archived = report_by_id[chosen_report_id]
            current_fingerprint = hashlib.sha256(source_log_text.encode("utf-8")).hexdigest() if source_log_text else None
            same_source = current_fingerprint and current_fingerprint == archived.get("source_fingerprint")
            st.session_state["agentguard_result"] = {
                "answer": archived.get("answer", ""), "status": archived.get("status", "completed"),
                "elapsed_seconds": archived.get("elapsed_seconds"), "steps": archived.get("steps", 0),
                "tool_calls": archived.get("tool_calls", 0),
                "trace": archived.get("tool_trace", []),
                "source_fingerprint": archived.get("source_fingerprint"),
            }
            st.session_state["agentguard_task"] = archived.get("task", "Saved investigation")
            st.session_state["agentguard_result_log"] = "logs.txt"
            st.session_state["agentguard_result_log_text"] = source_log_text if same_source else None
            st.session_state["agentguard_result_source_label"] = archived.get("source_label", "Saved report source")
            st.session_state["agentguard_report_id"] = archived.get("id")
            st.session_state["agentguard_archived_report"] = True
            st.session_state["agentguard_archived_metadata"] = {
                "build_id": archived.get("build_id", "Uploaded log"),
                "project_name": archived.get("project_name", "User supplied"),
                "build_status": archived.get("build_status", "Not provided"),
                "source_name": archived.get("source_name", "Saved report source"),
            }
            st.session_state["agentguard_page_target"] = "Investigations"
            st.rerun()
        st.caption("Reports and answers are stored locally with source citations. Uploaded log files themselves are not saved; re-upload the same log to reopen cited context.")

source_ready = source_log_text is not None
source_name = uploaded_name if source_log_text is not None else "No log uploaded"
archived_metadata = st.session_state.get("agentguard_archived_metadata") if st.session_state.get("agentguard_archived_report") else None
if archived_metadata:
    build_id = archived_metadata.get("build_id", build_id)
    project_name = archived_metadata.get("project_name", project_name)
    build_status = archived_metadata.get("build_status", build_status)
    source_name = archived_metadata.get("source_name", source_name)
if source_log_text is not None:
    source_size = len(source_log_text.encode("utf-8"))
    source_kind = f"{source_size / 1024:.1f} KB  /  Uploaded in this session"
    if archived_metadata:
        active_fingerprint = hashlib.sha256(source_log_text.encode("utf-8")).hexdigest()
        if active_fingerprint != st.session_state.get("agentguard_result", {}).get("source_fingerprint"):
            source_kind = "Saved report · original log not stored"
else:
    source_kind = "Saved report · original log not stored" if archived_metadata else "Waiting for upload"
source_indicator = "&#10003;" if source_ready else "!"
hero_col, source_col = st.columns([2.15, 1], gap="large", vertical_alignment="center")
with hero_col:
    st.markdown(
        '<section class="ag-hero"><div class="ag-kicker">AGENTGUARD / AI CI INVESTIGATOR</div>'
        '<h1>Investigate your <span style="color:#43bfff">CI failures</span></h1>'
        '<p>Upload a CI log, review the build context, and trace the root cause through evidence-backed insights.</p></section>',
        unsafe_allow_html=True,
    )
with source_col:
    st.markdown(
        '<div class="ag-source-card"><div class="ag-source-icon">&#128196;</div><div>'
        '<div class="ag-source-title">CI log source</div>'
        f'<div class="ag-source-name">{html.escape(str(source_name))}</div>'
        f'<div class="ag-source-detail">{html.escape(source_kind)}</div>'
        f'</div><div class="ag-source-ready">{source_indicator}</div></div>',
        unsafe_allow_html=True,
    )
st.markdown('<div class="ag-section-title">Build context</div>', unsafe_allow_html=True)
st.session_state["selected_log_filename"] = selected_log
summary_error = None
try:
    summary = get_test_summary(selected_log, source_log_text)
except (OSError, ValueError) as error:
    summary = {"available": False}
    summary_error = str(error)

if summary.get("available"):
    test_value = f'{summary["total"]} tests'
    test_detail = (
        f'{summary["passed"]} passed  /  {summary["failed"]} failed  /  '
        f'{summary["failure_rate_percent"]}% failure rate'
    )
else:
    test_value = "Unavailable"
    test_detail = (
        f"Test-count summary unavailable: {summary_error}"
        if summary_error else "No complete TOTAL / PASSED / FAILED summary in this log."
    )
status_class = "failed" if "fail" in str(build_status).lower() else "success" if str(build_status).lower() in {"passed", "success", "succeeded"} else ""
st.markdown(
    '<div class="ag-context-card">'
    f'<div class="ag-context-item"><span class="ag-context-label">Build</span><span class="ag-context-value">{html.escape(str(build_id))}</span></div>'
    f'<div class="ag-context-item"><span class="ag-context-label">Project</span><span class="ag-context-value">{html.escape(str(project_name))}</span></div>'
    f'<div class="ag-context-item"><span class="ag-context-label">Status</span><span class="ag-context-value ag-context-status {status_class}">{html.escape(str(build_status))}</span></div>'
    f'<div class="ag-context-item"><span class="ag-context-label">Test summary</span><span class="ag-context-value">{html.escape(test_value)}</span></div>'
    f'<div class="ag-context-note">{html.escape(test_detail)}</div>'
    '</div>',
    unsafe_allow_html=True,
)


if page == "Investigations":
    st.divider()
    st.session_state.setdefault("agentguard_history", [])

    history = st.session_state["agentguard_history"]
    if not history and not st.session_state.get("agentguard_result"):
        with st.container(border=True):
            st.markdown(
                '<div class="ag-prompt-heading"><div class="ag-prompt-spark">&#10022;</div><div>'
                '<h3>Start investigation</h3>'
                '<p>Choose a suggested question or ask your own. AgentGuard analyzes the log and returns source-backed findings.</p>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            prompt_cols = st.columns(3, gap="small")
            starter_prompts = [
                (
                    "Find the first failure",
                    "Identify the earliest failure, explain what happened, and point to the source evidence.",
                    "Find the first failure and explain its likely cause.",
                ),
                (
                    "Summarize the evidence",
                    "List failures, test results, and key findings with line references from the log.",
                    "Summarize the strongest evidence in this log.",
                ),
                (
                    "Check failure summaries",
                    "Verify test counts, failure rate, and available summaries in the log.",
                    "Check whether test counts or failure summaries are available.",
                ),
            ]
            for index, (col, (title, description, prompt)) in enumerate(zip(prompt_cols, starter_prompts)):
                with col:
                    with st.container(border=True):
                        st.markdown(
                            f'<div class="ag-prompt-card"><h4>{html.escape(title)}</h4>'
                            f'<p>{html.escape(description)}</p></div>',
                            unsafe_allow_html=True,
                        )
                        if st.button("Run investigation", use_container_width=True, key=f"starter_{index}"):
                            st.session_state["agentguard_suggested_prompt"] = prompt
                            st.rerun()

    task = st.chat_input(
        "Ask about this CI build...",
        key="investigation_chat",
    )
    if task is None:
        task = st.session_state.pop("agentguard_suggested_prompt", None)
    investigate = task is not None

    if investigate:
        if source_log_text is None:
            st.warning("Upload a valid text log before running an investigation.")
        elif not task.strip():
            st.warning("Enter an investigation request first.")
        else:
            previous = st.session_state.get("agentguard_result")
            previous_task = st.session_state.get("agentguard_task")
            if previous and previous_task:
                history = st.session_state.setdefault("agentguard_history", [])
                history.extend([
                    {"role": "user", "content": previous_task},
                    {"role": "assistant", "content": previous.get("answer", "")},
                ])
            context = ""
            if history:
                recent = history[-6:]
                context = "Previous conversation for context only:\n" + "\n".join(
                    f"{item['role'].title()}: {item['content']}" for item in recent
                ) + "\n\nCurrent request: "
            for key in (
                "agentguard_result", "agentguard_result_log", "agentguard_result_log_text",
                "agentguard_result_source_label", "agentguard_task",
            ):
                st.session_state.pop(key, None)
            agent_task = context + task
            progress_lines: list[str] = []
            with st.status("Preparing investigation…", expanded=True) as status:
                progress_view = st.empty()

                def report_progress(event: dict) -> None:
                    event_type = event.get("type")
                    if event_type == "plan":
                        progress_lines.append("Plan: " + " · ".join(event.get("items", [])))
                    elif event_type == "tool_complete":
                        mark = "⚠" if event.get("failed") else "✓"
                        progress_lines.append(f"{mark} Retrieved evidence with `{event.get('tool', 'tool')}`")
                    elif event_type == "model_step":
                        progress_lines.append(f"Working through model step {event.get('step', '?')}…")
                    progress_view.markdown("\n\n".join(progress_lines[-8:]))

                try:
                    result = run_agent(
                        agent_task,
                        progress_callback=report_progress,
                        log_filename=selected_log,
                        log_text=source_log_text,
                        source_label=selected_source_label if source_log_text is not None else None,
                    )
                    st.session_state["agentguard_result"] = result
                    st.session_state["agentguard_task"] = task
                    st.session_state["agentguard_result_log"] = selected_log
                    st.session_state["agentguard_result_log_text"] = source_log_text
                    st.session_state["agentguard_result_source_label"] = selected_source_label
                    report_id = str(uuid4())
                    st.session_state["agentguard_report_id"] = report_id
                    st.session_state["agentguard_archived_report"] = False
                    report_record = {
                        "id": report_id,
                        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                        "task": task,
                        "answer": result.get("answer", ""),
                        "status": result.get("status", "completed"),
                        "elapsed_seconds": result.get("elapsed_seconds"),
                        "steps": result.get("steps", 0),
                        "tool_calls": result.get("tool_calls", len(result.get("trace", []))),
                        "tool_trace": [
                            {
                                key: event[key]
                                for key in ("step", "tool", "arguments", "requested_by")
                                if key in event
                            }
                            | {
                                "line_numbers": sorted({
                                    int(match.group(1))
                                    for item in event.get("result", {}).get("result", [])
                                    if isinstance(item, str)
                                    for match in [re.match(r"L(\d+):", item)]
                                    if match
                                })
                            }
                            for event in result.get("trace", [])
                        ],
                        "source_name": uploaded_name or "uploaded log",
                        "source_label": selected_source_label,
                        "source_fingerprint": hashlib.sha256(source_log_text.encode("utf-8")).hexdigest(),
                        "build_id": build_id,
                        "project_name": project_name,
                        "build_status": build_status,
                    }
                    try:
                        save_report(report_record)
                    except OSError as error:
                        st.warning(f"Report is available for this session, but local history could not be saved: {error}")
                    status.update(label="Investigation complete", state="complete", expanded=False)
                except Exception as error:
                    status.update(label="Investigation failed", state="error", expanded=True)
                    st.error(f"The investigation could not be completed: {error}")
                    with st.expander("Technical details", expanded=True):
                        st.exception(error)


    # Render archived turns after processing the current input so the latest completed
    # turn never disappears during a follow-up run.
    for message in st.session_state.get("agentguard_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    result = st.session_state.get("agentguard_result")
    if result:
        result_log = st.session_state.get("agentguard_result_log", "logs.txt")
        result_log_text = st.session_state.get("agentguard_result_log_text")
        result_source_label = st.session_state.get("agentguard_result_source_label", f"data/{result_log}")
        with st.chat_message("user"):
            st.markdown(st.session_state.get("agentguard_task", "Investigate this build"))
        with st.chat_message("assistant"):
            st.markdown(result.get("answer", "No report was returned."))

        st.markdown("**Run another investigation**")
        followup_prompts = [
            ("Find first failure", "Find the first failure and explain its likely cause."),
            ("Summarize evidence", "Summarize the strongest evidence in this log."),
            ("Check test counts", "Check whether test counts or failure summaries are available."),
        ]
        followup_cols = st.columns(3, gap="small")
        for index, (col, (label, prompt)) in enumerate(zip(followup_cols, followup_prompts)):
            with col:
                if st.button(label, use_container_width=True, key=f"followup_{index}"):
                    st.session_state["agentguard_suggested_prompt"] = prompt
                    st.rerun()

        st.divider()
        report_col, meta_col = st.columns([3, 1], vertical_alignment="center")
        with report_col:
            st.markdown('<div class="ag-section"><span class="ag-step">3</span><span class="ag-section-title">Investigation report</span></div>', unsafe_allow_html=True)
        with meta_col:
            if result.get("status") == "completed":
                st.success("Complete", icon="✅")
            elif result.get("status") == "completed_with_fallback":
                st.warning("Evidence report · AI synthesis unavailable", icon="⚠️")
            elif result.get("status") == "incomplete_model_response":
                st.warning("AI response incomplete · output limit reached", icon="⚠️")
            else:
                st.warning(result.get("status", "Unknown status"), icon="⚠️")

        elapsed = result.get("elapsed_seconds")
        meta = []
        if elapsed is not None:
            meta.append(f"{elapsed:.2f}s")
        meta.append(f"{result.get('steps', 0)} model steps")
        meta.append(f"{result.get('tool_calls', len(result.get('trace', [])))} tool calls")
        st.markdown(
            '<div class="ag-report-meta">' + "".join(
                f'<span class="ag-chip">{item}</span>' for item in meta
            ) + '</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Model steps count Ollama turns. Tool calls count evidence tools run by AgentGuard; "
            "one model step can use several tool calls."
        )
        if result.get("status") == "completed_with_fallback" and result.get("model_error"):
            st.info(
                f"Ollama synthesis failed ({result['model_error']}). The report below is an "
                "evidence-based fallback, not an AI-written explanation. Check that Ollama is "
                "running and the selected model is available, then retry."
            )
        st.markdown('<div class="ag-label">Investigation details</div>', unsafe_allow_html=True)

        trace = result.get("trace", [])
        if result_log_text:
            timeline = log_timeline_events(result_log_text)
            with st.expander(f"Failure & retry timeline - {len(timeline)} events", expanded=False):
                if timeline:
                    st.dataframe(
                        [{"Line": number, "Event": kind, "Log entry": text}
                         for number, kind, text in timeline],
                        use_container_width=True, hide_index=True,
                        column_config={"Log entry": st.column_config.TextColumn(width="large")},
                    )
                else:
                    st.caption("No failure, retry, warning, or pipeline events were found in this log.")
        elif st.session_state.get("agentguard_archived_report"):
            st.info("This saved report does not include the source log. Upload the same file to view its timeline and line context.")

        st.markdown('<div class="ag-section" style="margin-top:1.5rem"><span class="ag-step">4</span><span class="ag-section-title">Source log evidence</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="ag-section-note">Select a line reference to inspect its surrounding source context.</div>', unsafe_allow_html=True)
        matches: set[tuple[str, int]] = set()
        for cited_number in re.findall(r"(?::L|\bL)(\d+)\b", result.get("answer", "")):
            matches.add((result_log, int(cited_number)))
        if not matches:
            for event in trace:
                matches.update(
                    (result_log, number)
                    for number in event.get("line_numbers", [])
                    if isinstance(number, int) and number > 0
                )
                if event.get("tool") != "search_logs":
                    continue
                payload = event.get("result", {}).get("result", [])
                if not isinstance(payload, list):
                    continue
                for item in payload:
                    if isinstance(item, str):
                        line_match = re.match(r"L(\d+):", item)
                        if line_match:
                            matches.add((result_log, int(line_match.group(1))))
        if matches:
            ordered_matches = sorted(matches)
            visible_matches = ordered_matches[:30]
            active_line = st.session_state.get("selected_evidence_line")
            if not any(line_number == active_line for _, line_number in visible_matches):
                active_line = visible_matches[0][1]
            for offset in range(0, len(visible_matches), 8):
                line_cols = st.columns(min(8, len(visible_matches) - offset), gap="small")
                for col, (filename, line_number) in zip(line_cols, visible_matches[offset:offset + 8]):
                    with col:
                        if st.button(
                            f"L{line_number}",
                            key=f"cite_{st.session_state.get('agentguard_report_id', 'current')}_{line_number}",
                            use_container_width=True,
                            type="primary" if line_number == active_line else "secondary",
                        ):
                            st.session_state["selected_evidence_line"] = line_number
                            st.rerun()
            if result_log_text:
                try:
                    context = get_log_context(result_log, active_line, radius=3, log_text=result_log_text)
                    st.caption(f"{result_source_label} - line {active_line}")
                    st.code("\n".join(
                        f"{'>' if row['matched'] else ' '} {row['line']:>6}  {row['text']}"
                        for row in context
                    ), language="text")
                except (OSError, ValueError) as error:
                    st.warning(f"Could not load this source context: {error}")
            else:
                st.caption("Upload the original log to inspect these saved line citations.")
        else:
            st.caption("No numbered source log lines were retrieved in this investigation.")
        if len(matches) > 30:
            st.caption(f"Showing 30 of {len(matches)} matching lines. The full results remain in the evidence trace.")

        tool_errors = [
            event for event in trace
            if isinstance(event.get("result"), dict) and "error" in event["result"]
        ]
        if tool_errors:
            st.warning(f"{len(tool_errors)} tool call(s) returned errors. Expand the trace for details.")

        tool_call_count = result.get("tool_calls", len(trace))
        with st.expander(f"Tools called · {tool_call_count} call(s)", expanded=True):
            if trace:
                call_rows = []
                for call_number, event in enumerate(trace, start=1):
                    tool_result = event.get("result", {})
                    payload = tool_result.get("result") if isinstance(tool_result, dict) else None
                    if "result" not in event:
                        outcome = "Tool name saved; evidence result not stored"
                    elif isinstance(tool_result, dict) and "error" in tool_result:
                        outcome = f"Error: {tool_result['error']}"
                    elif isinstance(payload, list):
                        found = [
                            item for item in payload
                            if isinstance(item, str) and not item.startswith("No matching")
                        ]
                        outcome = f"{len(found)} matching line(s)" if found else "No matches"
                    elif isinstance(payload, dict):
                        outcome = "Counts available" if payload.get("available") else "Summary returned"
                    else:
                        outcome = "Result returned"
                    call_rows.append({
                        "Call": call_number,
                        "Tool": event.get("tool", "Unknown tool"),
                        "Arguments": json.dumps(event.get("arguments", {}), ensure_ascii=False),
                        "Called by": "Harness" if event.get("requested_by") == "harness_followup" else "Model",
                        "Result": outcome,
                    })
                st.dataframe(call_rows, use_container_width=True, hide_index=True)
            else:
                if tool_call_count:
                    st.caption(
                        f"This saved report records {tool_call_count} tool calls, but their names "
                        "were not stored in the older report."
                    )
                else:
                    st.caption("No tool calls were recorded for this request.")

        if st.button("Clear report"):
            st.session_state.pop("agentguard_result", None)
            st.session_state.pop("agentguard_result_log", None)
            st.session_state.pop("agentguard_result_log_text", None)
            st.session_state.pop("agentguard_result_source_label", None)
            st.session_state.pop("agentguard_report_id", None)
            st.session_state.pop("agentguard_archived_report", None)
            st.session_state.pop("agentguard_archived_metadata", None)
            st.session_state.pop("selected_evidence_line", None)
            st.rerun()
elif page == "Overview":
    st.markdown("### Overview")
    with st.container(border=True):
        st.markdown("#### Current build")
        st.write(f"**{build_id}** / {project_name} / {build_status}")
        st.caption(f"Source: {source_name} / {source_kind}")
        if st.session_state.get("agentguard_result"):
            last_result = st.session_state["agentguard_result"]
            st.caption(f"Latest investigation: {last_result.get('status', 'unknown')} / {last_result.get('tool_calls', 0)} evidence calls")
        else:
            st.caption("No investigation has been run for this session yet.")
        if st.button("Open investigations", type="primary"):
            st.session_state["agentguard_page_target"] = "Investigations"
            st.rerun()
elif page == "Evidence":
    st.markdown("### Evidence")
    evidence_result = st.session_state.get("agentguard_result")
    if not evidence_result:
        st.info("Run an investigation to populate source evidence and the tool trace.")
    else:
        evidence_log = st.session_state.get("agentguard_result_log", selected_log)
        evidence_text = st.session_state.get("agentguard_result_log_text")
        evidence_source = st.session_state.get("agentguard_result_source_label", f"data/{evidence_log}")
        evidence_trace = evidence_result.get("trace", [])
        st.caption(f"Source: {evidence_source}")
        evidence_lines = {
            int(number) for number in re.findall(r"(?::L|\bL)(\d+)\b", evidence_result.get("answer", ""))
        }
        if not evidence_lines:
            for event in evidence_trace:
                evidence_lines.update(
                    number for number in event.get("line_numbers", [])
                    if isinstance(number, int) and number > 0
                )
                if event.get("tool") != "search_logs":
                    continue
                payload = event.get("result", {}).get("result", [])
                if isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, str):
                            line_match = re.match(r"L(\d+):", item)
                            if line_match:
                                evidence_lines.add(int(line_match.group(1)))
        if evidence_lines:
            for line_number in sorted(evidence_lines)[:50]:
                with st.expander(f"{evidence_source} / line {line_number}"):
                    try:
                        excerpt = get_log_context(evidence_log, line_number, radius=3, log_text=evidence_text)
                        st.code("\n".join(
                            f"{'>' if row['matched'] else ' '} {row['line']:>6}  {row['text']}"
                            for row in excerpt
                        ), language="text")
                    except (OSError, ValueError) as error:
                        st.warning(f"Could not load this source context: {error}")
        else:
            st.caption("No numbered log citations were returned for this investigation.")
        with st.expander(f"Evidence trace / {len(evidence_trace)} tool calls", expanded=False):
            if evidence_trace:
                st.dataframe(evidence_trace, use_container_width=True, hide_index=True)
                st.json(evidence_trace)
            else:
                st.caption("No tool calls were recorded.")
elif page == "Settings":
    st.markdown("### Settings")
    st.caption("Read-only runtime and data handling information for this workspace.")
    settings_cols = st.columns(2, gap="medium")
    with settings_cols[0]:
        with st.container(border=True):
            st.markdown("#### Model")
            st.code(MODEL, language=None)
            st.caption("Used for open-ended investigation tasks through the configured Ollama runtime.")
    with settings_cols[1]:
        with st.container(border=True):
            st.markdown("#### Data handling")
            st.write(f"Maximum uploaded log: {MAX_FILE_BYTES // (1024 * 1024)} MB")
            st.caption("Uploads remain in session memory. Bundled files are read-only; the agent tools do not modify project files.")
