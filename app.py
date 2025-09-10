# app.py — Streamlit UI for WAN 2.2 (via gradio_client) without st.secrets
# -----------------------------------------------------------------------------
# Usage:
#   python -m pip install streamlit gradio_client
#   export HF_TOKEN=hf_xxx        # (optional) only if the Space needs auth
#   python -m streamlit run app.py
# -----------------------------------------------------------------------------

import os
import re
import time
import shutil
import datetime
import traceback

import streamlit as st

# Try to import here so we can show a friendly UI error if missing.
try:
    from gradio_client import Client
except Exception as e:
    st.set_page_config(page_title="🎬 Text-to-Video Mini-Studio", layout="centered")
    st.title("🎬 Text-to-Video Mini-Studio")
    st.error(
        "The Python package `gradio_client` is not installed in this environment.\n\n"
        "Install it and re-run:\n\n"
        "```bash\n"
        "python -m pip install gradio_client streamlit\n"
        "python -m streamlit run app.py\n"
        "```"
    )
    st.stop()

# ---------------- Config ----------------
SPACE_IDS = [
    "zerogpu-aoti/wan2-2-fp8da-aoti",                   # owner/space id
    "https://zerogpu-aoti-wan2-2-fp8da-aoti.hf.space/", # direct URL
]

NEGATIVE_PROMPT = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, "
    "JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, "
    "形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"
)

PRESET_PROMPTS = {
    "Nail with hammer":
        "Demonstration of nailing a standard nail into a block of wood using a claw hammer.",
    "Screw in & out":
        "Demonstration of installing a screw into a block of wood using a power screwdriver, followed by uninstalling (unscrewing) the screw.",
    "Add gas to vehicle":
        "Adding gas from a gas can to a vehicle.",
}

OUT_DIR = "videos"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------- Helpers ----------------
def get_hf_token():
    """Prefer environment variable; do NOT use st.secrets to avoid local file requirement."""
    return os.getenv("HF_TOKEN")

def sanitize_filename(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_. -]+', '_', s)[:80]
    return s.strip().replace(' ', '_')

def connect(space_ids, hf_token=None) -> Client:
    """Try each Space form until one works; also verifies API."""
    last_err = None
    for target in space_ids:
        try:
            c = Client(target, hf_token=hf_token) if hf_token else Client(target)
            _ = c.view_api(return_format="dict")  # fail fast
            return c
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("Failed to connect to any Space target.")

def generate_video(
    client: Client,
    prompt: str,
    duration: float,
    guidance_scale: float,
    guidance_scale_2: float,
    steps: float,
    seed: float,
    randomize_seed: bool,
    retries: int = 4,
) -> tuple[str, float]:
    """
    Call the WAN 2.2 Space to generate a video.
    Returns: (saved_mp4_path, used_seed)
    Retries with shorter durations to dodge GPU quota/time limits.
    """
    durations_try = [duration, 5.0, 4.0, 3.5]
    last_exc = None

    for attempt, d in enumerate(durations_try[:retries], 1):
        try:
            res = client.predict(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                duration_seconds=float(d),
                guidance_scale=float(guidance_scale),
                guidance_scale_2=float(guidance_scale_2),
                steps=float(steps),
                seed=float(seed),
                randomize_seed=bool(randomize_seed),
                api_name="/generate_video",
            )
            out, used_seed = res
            video_path = out.get("video") if isinstance(out, dict) else None
            if not video_path or not os.path.exists(video_path):
                raise RuntimeError("Server returned but video file was not found.")

            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            basename = f"{ts}_{sanitize_filename(prompt[:60])}.mp4"
            dest = os.path.join(OUT_DIR, basename)
            shutil.copy(video_path, dest)
            return dest, float(used_seed)
        except Exception as e:
            last_exc = e
            # Gentle backoff between attempts
            time.sleep(1.2 * attempt)

    # All attempts failed, surface the last exception
    raise last_exc if last_exc else RuntimeError("Unknown generation failure.")

# ----------------- UI -------------------
st.set_page_config(page_title="🎬 Text-to-Video Mini-Studio", page_icon="🎬", layout="centered")
st.title("🎬 Text-to-Video Mini-Studio")
st.caption("WAN 2.2 Space via `gradio_client` • No `st.secrets` required")

with st.sidebar:
    st.header("Generation Settings")
    duration = st.slider("Duration (seconds)", 3.0, 6.0, 5.1, 0.1)
    steps = st.slider("Inference Steps", 2, 8, 4)
    g1 = st.slider("Guidance Scale (high-noise)", 0.5, 2.0, 1.0, 0.1)
    g2 = st.slider("Guidance Scale 2 (low-noise)", 1.0, 6.0, 3.0, 0.1)
    randomize_seed = st.checkbox("Randomize seed", True)
    seed = st.number_input("Seed", value=42, step=1)
    st.divider()
    st.caption("Tip: If the Space is gated/rate-limited, set HF_TOKEN as an environment variable.")

tab1, tab2 = st.tabs(["Presets", "Free-form"])

# Default prompt from presets
selected = st.radio("Choose a preset scene:", list(PRESET_PROMPTS.keys()), horizontal=True, key="preset_choice")
prompt = PRESET_PROMPTS[selected]

with tab2:
    free = st.text_area(
        "Describe your scene (free-form bonus)",
        height=120,
        placeholder="e.g., A close-up timelapse of a flower blooming on a wooden table.",
        key="freeform_text",
    )
    use_free = st.checkbox("Use free-form prompt instead of preset", key="use_freeform")
    if use_free and free.strip():
        prompt = free.strip()

hf_token = get_hf_token()
if hf_token is None:
    st.info("No HF token detected in environment. If the Space is public, you're fine. "
            "Otherwise, set it with:\n\n`export HF_TOKEN=hf_xxx`", icon="ℹ️")

@st.cache_resource
def cached_client(token):
    # token is part of the cache key; if you set/clear HF_TOKEN, the client will refresh
    return connect(SPACE_IDS, token)

# Try to create the client up-front so the "Generate" button is fast
try:
    client = cached_client(hf_token)
except Exception as e:
    st.error("Failed to connect to the WAN 2.2 Space.\n\n"
             f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```")
    st.stop()

if st.button("Generate 🎥", use_container_width=True):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()
    with st.spinner("Generating video..."):
        try:
            path, used_seed = generate_video(client, prompt, duration, g1, g2, steps, seed, randomize_seed)
        except Exception as e:
            st.error("Generation failed.\n\n"
                     f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```")
            st.stop()

    st.success("Done!")
    st.video(path)
    with open(path, "rb") as f:
        st.download_button("Download .mp4", f, file_name=os.path.basename(path), mime="video/mp4")
    st.caption(f"Seed: {used_seed} • Saved to: `{path}`")

st.markdown(
    """
---
**Notes**
- If you hit GPU quota/time limits on the Space, the app auto-retries with shorter durations.
- For reproducible results, uncheck **Randomize seed**.
- Outputs are saved under the local `videos/` folder.
"""
)
