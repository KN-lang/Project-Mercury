"""
app.py — Project Mercury
Streamlit frontend: audio input → STT → intent → tool execution → display.
"""
import io
import logging
import time

import streamlit as st

from config import (
    APP_NAME, APP_EMOJI, APP_VERSION,
    CONFIRMATION_REQUIRED, LANG_EXTENSION_MAP, OUTPUT_DIR,
)
from memory import (
    init_memory, add_to_history, get_history, clear_history,
    set_pending, clear_pending,
)
from tools import lang_to_extension

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title=f"{APP_NAME} — Voice AI Agent",
    page_icon=APP_EMOJI,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & Background ── */
html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #080c14;
    color: #c8d0e0;
}
.stApp {
    background: radial-gradient(ellipse at 20% 0%, rgba(120,80,255,0.10) 0%, transparent 55%),
                radial-gradient(ellipse at 80% 100%, rgba(200,169,126,0.07) 0%, transparent 50%),
                #080c14;
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,15,25,0.95) !important;
    border-right: 1px solid rgba(200,169,126,0.12) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c8a97e;
}

/* ── Mercury header ── */
.mercury-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 24px 0 8px;
    border-bottom: 1px solid rgba(200,169,126,0.15);
    margin-bottom: 28px;
}
.mercury-logo {
    font-size: 3rem;
    line-height: 1;
    filter: drop-shadow(0 0 16px rgba(200,169,126,0.6));
}
.mercury-title { margin: 0; }
.mercury-title h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #e8c97a 0%, #c8a97e 50%, #a07848 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}
.mercury-title p {
    margin: 2px 0 0;
    font-size: 0.85rem;
    color: #60687a;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Cards ── */
.m-card {
    background: rgba(255,255,255,0.032);
    border: 1px solid rgba(200,169,126,0.12);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 14px 0;
    transition: border-color 0.2s;
}
.m-card:hover { border-color: rgba(200,169,126,0.22); }

.m-card-success {
    background: rgba(16,185,129,0.07);
    border-color: rgba(16,185,129,0.25);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 14px 0;
}
.m-card-warning {
    background: rgba(245,158,11,0.07);
    border-color: rgba(245,158,11,0.28);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 14px 0;
}
.m-card-error {
    background: rgba(239,68,68,0.07);
    border-color: rgba(239,68,68,0.25);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 14px 0;
}

/* ── Section labels ── */
.step-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #c8a97e;
    margin-bottom: 6px;
}

/* ── Intent badge ── */
.intent-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.badge-CHAT        { background: rgba(59,130,246,0.18); color:#60a5fa; border:1px solid rgba(59,130,246,0.30); }
.badge-CREATE_FILE { background: rgba(16,185,129,0.18); color:#34d399; border:1px solid rgba(16,185,129,0.30); }
.badge-WRITE_CODE  { background: rgba(139,92,246,0.18); color:#a78bfa; border:1px solid rgba(139,92,246,0.30); }
.badge-SUMMARIZE   { background: rgba(245,158,11,0.18); color:#fbbf24; border:1px solid rgba(245,158,11,0.30); }

/* ── History item ── */
.history-item {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 12px 14px;
    margin: 8px 0;
    font-size: 0.82rem;
}
.history-ts { color: #40485a; font-size: 0.72rem; margin-bottom: 4px; }
.history-text { color: #8090a8; margin: 4px 0; }
.history-output { color: #c8d0e0; margin-top: 6px; }

/* ── Monospace output ── */
.mono-output {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    background: rgba(0,0,0,0.25);
    border-radius: 8px;
    padding: 14px;
    color: #a0e0c0;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Divider ── */
.m-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 20px 0;
}

/* ── Confidence meter ── */
.conf-bar-wrap { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
.conf-bar { flex: 1; height: 4px; background: rgba(255,255,255,0.08); border-radius: 2px; overflow: hidden; }
.conf-fill { height:100%; border-radius:2px; background: linear-gradient(90deg,#c8a97e,#e8c97a); transition: width 0.4s; }
.conf-pct  { font-size: 0.75rem; color: #c8a97e; min-width: 36px; text-align: right; }

/* ── Streamlit button overrides ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600;
    color: #60687a;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #c8a97e !important;
    border-bottom-color: #c8a97e !important;
}

/* ── Spinner tint ── */
.stSpinner > div > div { border-top-color: #c8a97e !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached singletons (loaded once per Streamlit session) ─────────────────────
@st.cache_resource(show_spinner="☿ Loading Whisper … (first run downloads model)")
def _load_audio_processor():
    from audio_processor import AudioProcessor
    return AudioProcessor()


@st.cache_resource(show_spinner="☿ Connecting to Ollama …")
def _load_agent():
    from agent import MercuryAgent
    return MercuryAgent()


@st.cache_resource
def _load_executor():
    from tools import ToolExecutor
    return ToolExecutor()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _intent_badge(intent: str) -> str:
    icons = {
        "CHAT":        "💬",
        "CREATE_FILE": "📄",
        "WRITE_CODE":  "💻",
        "SUMMARIZE":   "📝",
    }
    icon = icons.get(intent, "❓")
    return f'<span class="intent-badge badge-{intent}">{icon} {intent.replace("_"," ")}</span>'


def _confidence_bar(confidence: float) -> str:
    pct = int(confidence * 100)
    return (
        f'<div class="conf-bar-wrap">'
        f'  <span style="font-size:0.75rem;color:#60687a;">Confidence</span>'
        f'  <div class="conf-bar"><div class="conf-fill" style="width:{pct}%"></div></div>'
        f'  <span class="conf-pct">{pct}%</span>'
        f'</div>'
    )


def _convert_to_wav(uploaded_file) -> bytes:
    """
    Convert any UploadedFile (wav/mp3/webm/ogg) to WAV bytes via pydub.
    Requires ffmpeg on PATH (installed via brew).
    """
    try:
        from pydub import AudioSegment
        ext = (uploaded_file.name.rsplit(".", 1)[-1].lower()
               if hasattr(uploaded_file, "name") and "." in getattr(uploaded_file, "name", "")
               else "wav")
        audio = AudioSegment.from_file(io.BytesIO(uploaded_file.read()), format=ext)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        return wav_io.getvalue()
    except Exception as exc:
        # If pydub/ffmpeg unavailable, pass raw bytes and hope Whisper can handle it
        logger.warning(f"pydub conversion failed ({exc}), passing raw bytes to Whisper")
        uploaded_file.seek(0)
        return uploaded_file.read()


# ── Pipeline execution ────────────────────────────────────────────────────────

def _run_pipeline(audio_source) -> None:
    """
    Full Mercury pipeline:
      1. Transcribe audio  →  2. Classify intent  →  3. Gate (HITL) or execute
    """
    processor = _load_audio_processor()
    agent     = _load_agent()

    # ── Step 1: STT ───────────────────────────────────────────────────────────
    with st.spinner("🎙️ Transcribing audio …"):
        transcription = processor.transcribe(audio_source)

    if not transcription:
        st.session_state.last_result = {
            "success": False,
            "intent":  "—",
            "transcription": "",
            "output": (
                "Mercury couldn't make out any speech in that audio. "
                "Please try speaking more clearly, or check your microphone level."
            ),
        }
        return

    st.session_state.transcription = transcription

    # ── Step 2: Intent clasification ──────────────────────────────────────────
    with st.spinner("🧠 Understanding your intent …"):
        intent_result = agent.classify_intent(transcription)

    st.session_state.intent_result = intent_result
    intent = intent_result.get("intent", "CHAT")

    # ── Step 3: Route ─────────────────────────────────────────────────────────
    if intent in CONFIRMATION_REQUIRED:
        # Park and wait for human approval
        set_pending({
            "intent":       intent,
            "intent_result": intent_result,
            "transcription": transcription,
        })
    else:
        _execute_action(intent, intent_result, transcription)


def _execute_action(intent: str, intent_result: dict, transcription: str) -> None:
    """Execute a classified action; update last_result and history."""
    agent    = _load_agent()
    executor = _load_executor()
    history  = get_history()

    output  = ""
    success = True

    if intent == "CHAT":
        with st.spinner("☿ Mercury is thinking …"):
            output = agent.chat(transcription, history)

    elif intent == "SUMMARIZE":
        text_to_summarize = intent_result.get("text") or transcription
        with st.spinner("📝 Summarizing …"):
            output = agent.summarize(text_to_summarize)

    elif intent == "CREATE_FILE":
        filename = intent_result.get("filename") or "mercury_note.txt"
        content  = intent_result.get("content") or ""
        success, output = executor.create_file(filename, content)

    elif intent == "WRITE_CODE":
        language = (intent_result.get("language") or "python").lower()
        task     = intent_result.get("task") or intent_result.get("content") or transcription
        filename = intent_result.get("filename") or f"mercury_code.{lang_to_extension(language)}"

        with st.spinner(f"💻 Generating {language} code …"):
            code = agent.generate_code(task, language)

        success, output = executor.write_code(filename, code, language)
        if success:
            output += f"\n\n```{language}\n{code}\n```"

    else:
        output  = "Mercury encountered an unmapped intent. Could you rephrase that?"
        success = False

    result = {
        "success":      success,
        "intent":       intent,
        "transcription": transcription,
        "output":        output,
        "confidence":    intent_result.get("confidence", 0.5),
    }
    st.session_state.last_result = result
    add_to_history(result)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px;">
            <div style="font-size:2.4rem;filter:drop-shadow(0 0 12px rgba(200,169,126,0.5))">☿</div>
            <div style="font-size:1.1rem;font-weight:700;color:#c8a97e;letter-spacing:-0.01em">MERCURY</div>
            <div style="font-size:0.68rem;color:#40485a;letter-spacing:0.12em;text-transform:uppercase">Voice AI Agent</div>
        </div>
        <hr style="border:none;border-top:1px solid rgba(200,169,126,0.12);margin:12px 0 20px;">
        """, unsafe_allow_html=True)

        # ── System status ──
        st.markdown('<p class="step-label">System Status</p>', unsafe_allow_html=True)

        # Ollama status
        try:
            import ollama as _ol
            _ol.list()
            st.success("🟢 Ollama  connected", icon=None)
        except Exception:
            st.error("🔴 Ollama  offline — run `ollama serve`")

        # MLX Whisper
        try:
            import mlx_whisper  # noqa
            st.success("🟢 Whisper  MLX (Apple Silicon)", icon=None)
        except ImportError:
            try:
                import whisper  # noqa
                st.warning("🟡 Whisper  CPU fallback")
            except ImportError:
                st.error("🔴 Whisper  not installed")

        st.markdown("<hr class='m-divider'>", unsafe_allow_html=True)

        # ── Output folder ──
        st.markdown('<p class="step-label">Output Files</p>', unsafe_allow_html=True)
        try:
            executor = _load_executor()
            files = executor.list_files()
            if files:
                for f in files:
                    st.markdown(f"📄 `{f}`")
            else:
                st.caption("No files yet — ask Mercury to create one!")
        except Exception:
            st.caption("(executor not ready)")

        st.markdown("<hr class='m-divider'>", unsafe_allow_html=True)

        # ── History ──
        st.markdown('<p class="step-label">Session History</p>', unsafe_allow_html=True)
        history = get_history()
        if not history:
            st.caption("No actions yet this session.")
        else:
            for entry in reversed(history[-10:]):   # newest first, max 10
                intent = entry.get("intent", "CHAT")
                icon   = {"CHAT":"💬","CREATE_FILE":"📄","WRITE_CODE":"💻","SUMMARIZE":"📝"}.get(intent,"❓")
                ts     = entry.get("timestamp","")
                text   = (entry.get("transcription","") or "")[:55]
                ellip  = "…" if len(entry.get("transcription","") or "") > 55 else ""
                st.markdown(
                    f'<div class="history-item">'
                    f'  <div class="history-ts">{ts}</div>'
                    f'  <div>{icon} <strong style="color:#c8a97e">{intent.replace("_"," ")}</strong></div>'
                    f'  <div class="history-text">"{text}{ellip}"</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.button("🗑️ Clear history", on_click=clear_history, use_container_width=True)


# ── Main UI ───────────────────────────────────────────────────────────────────

def main() -> None:
    init_memory()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="mercury-header">
        <div class="mercury-logo">☿</div>
        <div class="mercury-title">
            <h1>Mercury</h1>
            <p>Local Voice AI Agent · v{ver}</p>
        </div>
    </div>
    """.format(ver=APP_VERSION), unsafe_allow_html=True)

    _render_sidebar()

    # ── Two-column layout ─────────────────────────────────────────────────────
    left_col, right_col = st.columns([1.15, 1], gap="large")

    # ════════════════════════════════════════════════════════════════════════════
    # LEFT: AUDIO INPUT
    # ════════════════════════════════════════════════════════════════════════════
    with left_col:
        st.markdown('<p class="step-label">🎙️ Audio Input</p>', unsafe_allow_html=True)

        mic_tab, upload_tab = st.tabs(["🎤  Record (primary)", "📂  Upload file"])

        audio_to_process = None   # will hold bytes or file-like

        # ── Microphone tab ────────────────────────────────────────────────────
        with mic_tab:
            st.markdown(
                '<div class="m-card">'
                '<strong style="color:#c8d0e0">Live microphone</strong><br/>'
                '<span style="font-size:0.82rem;color:#60687a">Click the mic icon below, speak, then click stop.'
                ' Mercury will process your voice immediately.</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            mic_audio = st.audio_input(
                "Record your message",
                key="mic_input",
                label_visibility="collapsed",
            )
            if mic_audio is not None:
                audio_to_process = mic_audio

        # ── Upload tab ────────────────────────────────────────────────────────
        with upload_tab:
            st.markdown(
                '<div class="m-card">'
                '<strong style="color:#c8d0e0">Upload audio file</strong><br/>'
                '<span style="font-size:0.82rem;color:#60687a">Supported: .wav, .mp3, .ogg, .webm</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Choose audio file",
                type=["wav", "mp3", "ogg", "webm", "m4a"],
                key="file_upload",
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                st.audio(uploaded_file)
                audio_to_process = uploaded_file

        # ── Process button ────────────────────────────────────────────────────
        st.markdown("<br/>", unsafe_allow_html=True)
        if audio_to_process is not None:
            if st.button(
                "⚡  Process with Mercury",
                key="btn_process",
                use_container_width=True,
                type="primary",
            ):
                # Reset pending state for a fresh run
                clear_pending()
                st.session_state.last_result   = None
                st.session_state.intent_result = None
                st.session_state.transcription = ""

                try:
                    _run_pipeline(audio_to_process)
                except ConnectionError as ce:
                    st.error(f"**Ollama not reachable.** {ce}", icon="🔴")
                except RuntimeError as re_:
                    st.error(f"**Whisper error.** {re_}", icon="🎙️")
                except Exception as exc:
                    logger.error("Pipeline error", exc_info=True)
                    st.error(f"Mercury encountered an unexpected error: {exc}", icon="⚠️")
                st.rerun()
        else:
            st.info("Record or upload audio above, then click **Process with Mercury**.")

    # ════════════════════════════════════════════════════════════════════════════
    # RIGHT: PIPELINE RESULTS
    # ════════════════════════════════════════════════════════════════════════════
    with right_col:
        st.markdown('<p class="step-label">🔬 Pipeline Output</p>', unsafe_allow_html=True)

        # ── Human-in-the-loop gate ────────────────────────────────────────────
        if st.session_state.awaiting_confirmation:
            pending = st.session_state.pending_action or {}
            intent  = pending.get("intent", "ACTION")
            ir      = pending.get("intent_result", {})
            trans   = pending.get("transcription", "")

            st.markdown(f"""
            <div class="m-card-warning">
                <p class="step-label">⚠️ Human-in-the-Loop Confirmation</p>
                <p style="color:#c8d0e0;margin:8px 0 4px">
                    Mercury wants to perform a <strong>{intent.replace('_',' ')}</strong> operation.
                </p>
                <p class="history-text">You said: <em>"{trans}"</em></p>
            </div>
            """, unsafe_allow_html=True)

            # Show what will be done
            filename = ir.get("filename") or "(auto-named)"
            lang     = ir.get("language") or ""
            st.markdown(f"**Action:** `{intent}` &nbsp;|&nbsp; **File:** `{filename}`" +
                        (f" &nbsp;|&nbsp; **Language:** `{lang}`" if lang else ""))

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅  Yes — Execute", key="hitl_yes", use_container_width=True, type="primary"):
                    act_data = pending.copy()
                    clear_pending()
                    _execute_action(act_data["intent"], act_data["intent_result"], act_data["transcription"])
                    st.rerun()
            with c2:
                if st.button("❌  No — Cancel", key="hitl_no", use_container_width=True):
                    clear_pending()
                    st.session_state.last_result = {
                        "success":      False,
                        "intent":       intent,
                        "transcription": trans,
                        "output":       "Mercury cancelled the action as requested. No files were modified.",
                        "confidence":   ir.get("confidence", 0.5),
                    }
                    add_to_history(st.session_state.last_result)
                    st.rerun()

            st.stop()   # Don't render pipeline steps while waiting for confirmation

        # ── Step display: Transcription ───────────────────────────────────────
        transcription = st.session_state.get("transcription", "")

        if transcription:
            st.markdown("""
            <div class="m-card">
                <p class="step-label">📝 Step 1 — Transcription (Whisper)</p>
            """, unsafe_allow_html=True)
            st.markdown(f"> {transcription}")
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Step display: Intent ──────────────────────────────────────────
            ir = st.session_state.get("intent_result")
            if ir:
                intent     = ir.get("intent", "CHAT")
                confidence = ir.get("confidence", 0.5)
                st.markdown(f"""
                <div class="m-card">
                    <p class="step-label">🧠 Step 2 — Intent Detection (Ollama · qwen2.5:0.5b)</p>
                    <div style="margin:6px 0">{_intent_badge(intent)}</div>
                    {_confidence_bar(confidence)}
                </div>
                """, unsafe_allow_html=True)

            # ── Step display: Result ──────────────────────────────────────────
            result = st.session_state.get("last_result")
            if result:
                intent  = result.get("intent", "CHAT")
                success = result.get("success", True)
                output  = result.get("output", "")

                card_cls = "m-card-success" if success else "m-card-error"
                status   = "✅ Action Completed" if success else "⚠️ Action Failed"
                action_label = {
                    "CHAT":        "Mercury replied",
                    "CREATE_FILE": "File created",
                    "WRITE_CODE":  "Code generated & saved",
                    "SUMMARIZE":   "Summary produced",
                }.get(intent, "Result")

                st.markdown(f"""
                <div class="{card_cls}">
                    <p class="step-label">⚡ Step 3 — {action_label}</p>
                    <p style="margin:0 0 4px;font-size:0.8rem;color:#60687a">{status}</p>
                </div>
                """, unsafe_allow_html=True)

                # Render markdown output (supports code fences)
                st.markdown(output)

        else:
            # Welcome state
            st.markdown("""
            <div class="m-card" style="text-align:center;padding:40px 24px;">
                <div style="font-size:2.8rem;margin-bottom:12px;filter:drop-shadow(0 0 20px rgba(200,169,126,0.4))">☿</div>
                <p style="font-size:1.05rem;font-weight:600;color:#c8a97e;margin:0 0 8px">Hello, I'm Mercury.</p>
                <p style="color:#60687a;font-size:0.88rem;margin:0">
                    Record your voice or upload an audio file.<br/>
                    I can <strong style="color:#a78bfa">write code</strong>,
                    <strong style="color:#34d399">create files</strong>,
                    <strong style="color:#fbbf24">summarize text</strong>,
                    or just <strong style="color:#60a5fa">chat</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
