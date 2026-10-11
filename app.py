"""
EchoSign - two-way emergency medical translation bridge (HackRU)


Run:  streamlit run app.py
Needs: key.txt (Gemini API key, first line) next to app.py, and an ./assets folder.
"""


import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


import av
import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer


# ---------------------------------------------------------------------------
# Streamlit page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="EchoSign", page_icon="🤟", layout="wide")


# ---------------------------------------------------------------------------
# Optional third-party imports (shown as friendly errors instead of crashing)
# ---------------------------------------------------------------------------
try:
   from google import genai as genai_new


   GENAI_MODE = "new"
except ImportError:
   try:
       import google.generativeai as genai_old


       GENAI_MODE = "legacy"
   except ImportError:
       GENAI_MODE = None


try:
   from elevenlabs.client import ElevenLabs
   from elevenlabs import play as eleven_play


   ELEVEN_OK = True
except ImportError:
   ELEVEN_OK = False


try:
   import speech_recognition as sr


   SR_OK = True
except ImportError:
   SR_OK = False


try:
   import pyaudio  # noqa: F401


   PYAUDIO_OK = True
except Exception:  # noqa: BLE001
   PYAUDIO_OK = False


try:
   import sounddevice as sd


   SD_OK = True
except Exception:  # noqa: BLE001
   SD_OK = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
KEY_FILE = BASE_DIR / "key.txt"
# Never hardcode API secrets in source code. Set ELEVENLABS_API_KEY in your shell.
ELEVEN_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "").strip()
ELEVEN_BASE = "https://api.elevenlabs.io/v1"
# Fallback list. The app first asks Google which Flash models your key can use
# and tries those newest-first; these are only added as a safety net.
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]
SAMPLE_INTERVAL_SECONDS = 0.5  # analyze promptly when the previous Gemini call finishes
HEARTBEAT_SECONDS = 0.8       # retry quickly even when the camera view is mostly still
MOTION_THRESHOLD = 0.8          # detect small hand movements sooner
REPEAT_SPEAK_SECONDS = 8.0      # re-announce a gesture that is still being held
CONFIRMATION_COUNT = 1          # respond on the first clear Gemini prediction to reduce delay
CONFIRMATION_WINDOW_SECONDS = 3.0   # short candidate window
SPEAK_GESTURES = tuple()  # Spoken output is driven by the supported gesture phrase map below.
MIC_SECONDS = 4
MIC_SILENT_LEVEL = 150          # peak below this means the mic is giving us silence
BUSY_TIMEOUT_SECONDS = 15.0     # recover sooner if a Gemini call hangs
ASSET_EXTENSIONS = (".mp4", ".gif", ".mov", ".webm", ".m4v")


GEMINI_PROMPT = (
   "You are a best-effort camera classifier for a medical ASL communication demo. "
   "Inspect the visible hands and body pose in this camera frame. Return exactly ONE "
   "label from the supported vocabulary, never a sentence or combination of labels. "
   "If no hands are clearly visible, return None. If the person is resting, sitting still, "
   "or has a hand casually in view without a clearly recognizable ASL sign, return None. "
   "Do not assume the person is signing just because a hand is visible. For Hello, require "
   "a clear greeting wave or recognizable raised-hand greeting near the head; an arbitrary "
   "open hand is NOT Hello. "
   "For Help, look specifically for the ASL help configuration: one hand held palm-up horizontally "
   "with the other hand in a fist resting on that palm; the sign often lifts upward. Return Help "
   "only when this hand configuration is clearly visible; do not guess from a random raised hand. "
   "Use only visible hand shape and position. If the hands are not visible or there is no clear "
   "evidence for a supported sign, return None. "
   "Never guess a medical phrase based on context, emotion, or what seems medically likely. "
   "A single still frame cannot reliably distinguish many ASL signs, especially signs that "
   "depend on movement or context; this is a demo, not a substitute for a qualified ASL interpreter.\n"
   "Supported labels: Hello, Help, Please, ThankYou, Yes, No, Maybe, DontKnow, Agree, "
   "Disagree, Understand, DontUnderstand, NeedInterpreter, DeafHardOfHearing, NeedHelp, "
   "NeedPenPaper, Stop, Wait, Pain, HeadPain, StomachPain, ChestPain, BackPain, ArmPain, "
   "LegPain, Dizzy, Nauseous, ShortOfBreath, Weak, Fever, Sick, Better, Worse, MildPain, "
   "SeverePain, Water, Bathroom, SitDown, Allergy, Medication, MedicalCondition, Diabetes, "
   "Asthma, FamilyMedicalHistory, NeedDoctor, NeedNurse, WhatWrong, WhatHappened, "
   "WhatMedicine, MedicinePurpose, WillItHurt, WhatShouldIDo, ExplainAgain, HaveQuestion, "
   "WriteItDown, Hungry, Thirsty, Tired, Cold, Hot, Scared, Worried, Sorry, ExcuseMe, "
   "GoodMorning, GoodNight, Repeat, SlowDown, NeedTranslator, NeedPrivacy, CallFamily, "
   "Emergency, NotOkay, FeelingOkay, WhereBathroom, A little, A lot, Now, Later, Before, After, None.\n"
   "Do not infer symptoms from facial expression alone. For symptom labels, use only a "
   "clearly recognizable sign. Also estimate visible facial expression separately as one word: "
   "Happy, Panicked, Scared, Calm, Sad, Angry or Neutral. Do not use emotion to infer gesture.\n"
   "Respond STRICTLY with a raw JSON object and nothing else, using exactly these keys: "
   "{\"gesture\": \"one supported label\", \"emotion\": \"one word\"}"
)


SPOKEN_PHRASES = {
   # Original two phrases are retained.
   "Hello": "Hello.",
   "Help": "Help! Please, I need help right now!",
   "Please": "Please.",
   "ThankYou": "Thank you.",
   "Yes": "Yes.",
   "No": "No.",
   "Maybe": "Maybe.",
   "DontKnow": "I don't know.",
   "Agree": "I agree.",
   "Disagree": "I disagree.",
   "Understand": "I understand.",
   "DontUnderstand": "I don't understand.",
   "NeedInterpreter": "I am Deaf or hard of hearing. I need a qualified ASL interpreter, please.",
   "DeafHardOfHearing": "I am Deaf or hard of hearing.",
   "NeedHelp": "I need help, please.",
   "NeedPenPaper": "Please give me a pen and paper, or let me type on a phone.",
   "Stop": "Please stop.",
   "Wait": "Please wait.",
   "Pain": "I am experiencing pain.",
   "HeadPain": "My head hurts.",
   "StomachPain": "My stomach hurts.",
   "ChestPain": "My chest hurts. Please assess me.",
   "BackPain": "My back hurts.",
   "ArmPain": "My arm hurts.",
   "LegPain": "My leg hurts.",
   "Dizzy": "I feel dizzy.",
   "Nauseous": "I feel nauseous.",
   "ShortOfBreath": "I am short of breath. Please help me.",
   "Weak": "I feel weak.",
   "Fever": "I may have a fever.",
   "Sick": "I feel sick.",
   "Better": "I feel better.",
   "Worse": "I feel worse.",
   "MildPain": "My pain is mild.",
   "SeverePain": "My pain is severe.",
   "Water": "I need water, please.",
   "Bathroom": "I need to use the bathroom, please.",
   "SitDown": "I need to sit down, please.",
   "Allergy": "I have an allergy. Please ask me what I am allergic to.",
   "Medication": "I have a question about my medication.",
   "MedicalCondition": "I have a medical condition. Please ask me about it.",
   "Diabetes": "I have diabetes.",
   "Asthma": "I have asthma.",
   "FamilyMedicalHistory": "Please ask me about my family medical history.",
   "NeedDoctor": "I need to speak with a doctor.",
   "NeedNurse": "I need a nurse, please.",
   "WhatWrong": "Please explain what is wrong.",
   "WhatHappened": "What happened?",
   "WhatMedicine": "What medicine is this?",
   "MedicinePurpose": "What does this medicine do?",
   "WillItHurt": "Will this hurt?",
   "WhatShouldIDo": "What should I do?",
   "ExplainAgain": "Please explain that again.",
   "HaveQuestion": "I have a question.",
   "WriteItDown": "Please write it down for me.",
   "Hungry": "I am hungry.",
   "Thirsty": "I am thirsty.",
   "Tired": "I am tired.",
   "Cold": "I feel cold.",
   "Hot": "I feel hot.",
   "Scared": "I feel scared.",
   "Worried": "I feel worried.",
   "Sorry": "I am sorry.",
   "ExcuseMe": "Excuse me, please.",
   "GoodMorning": "Good morning.",
   "GoodNight": "Good night.",
   "Repeat": "Please repeat that.",
   "SlowDown": "Please speak more slowly.",
   "NeedTranslator": "I need a translator, please.",
   "NeedPrivacy": "I need some privacy, please.",
   "CallFamily": "Please contact my family.",
   "Emergency": "This is an emergency. Please get medical help.",
   "NotOkay": "I do not feel okay.",
   "FeelingOkay": "I feel okay.",
   "WhereBathroom": "Where is the bathroom?",
   "A little": "A little.",
   "A lot": "A lot.",
   "Now": "Now, please.",
   "Later": "Later, please.",
   "Before": "Before.",
   "After": "After.",
}


SIGN_DICTIONARY = [
   (("hello", "hi"), "assets/hello.mp4"),
   (("help", "assistance"), "assets/help.gif"),
   (("doctor", "medic", "physician"), "assets/doctor-medic.gif"),
   (("pain", "hurt"), "assets/hurt-01.mp4"),
   (("thank you", "thanks"), "assets/thank-you.gif"),
   (("nurse",), "assets/nurse.mp4"),
   (("wait", "please wait", "hold on"), "assets/wait.gif"),
]


# ---------------------------------------------------------------------------
# Dark theme CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
   background-color: #0d1117 !important;
   color: #c9d1d9;
}
.block-container { padding-top: 2rem; }
.echo-title, p.echo-title {
   font-size: clamp(5.5rem, 22vw, 15rem) !important;
   font-weight: 900 !important;
   line-height: 1.08 !important;
   letter-spacing: -0.055em !important;
   margin: 0 0 0.15rem 0 !important;
   padding: 0.05em 0 0.12em !important;
   display: block !important;
   text-align: center !important;
   width: 100% !important;
   background: linear-gradient(90deg, #79c0ff 0%, #d2a8ff 52%, #ffa198 100%);
   -webkit-background-clip: text !important;
   background-clip: text !important;
   -webkit-text-fill-color: transparent !important;
   color: #d2a8ff !important;
   filter: drop-shadow(0 2px 12px rgba(121, 192, 255, 0.14));
}
.echo-subtitle { text-align: center !important; color: #c9d1d9; margin: -0.05rem auto 1.2rem !important; font-size: clamp(1.5rem, 2.2vw, 1.7rem) !important; line-height: 1.35; font-weight: 500; max-width: 1100px; width: 100%; display: block; }
div[data-testid="column"], div[data-testid="stColumn"] {
   background-color: #161b22;
   border: 1px solid #30363d;
   border-radius: 14px;
   padding: 1.2rem 1.4rem !important;
}
.card-title { font-size: 1.35rem; font-weight: 700; color: #f0f6fc; margin-bottom: 0.2rem; }
.card-caption { color: #aebdca; font-size: 0.9rem; margin-bottom: 0.8rem; }
/* Make the typed-phrase prompt brighter and more legible */
div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] label p {
   color: #e6edf3 !important;
   font-weight: 600 !important;
}
div[data-testid="stTextInput"] input::placeholder {
   color: #9da7b1 !important;
   opacity: 1 !important;
}
[data-testid="stMetric"] {
   background-color: #0d1117;
   border: 1px solid #30363d;
   border-radius: 10px;
   padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] p { color: #8b949e !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
   color: #ff8c1a !important;
   font-weight: 800;
}
.stButton > button {
   width: 100%;
   background-color: #21262d;
   color: #f0f6fc;
   border: 1px solid #ff8c1a;
   border-radius: 10px;
   padding: 0.7rem 1rem;
   font-weight: 700;
}
.stButton > button:hover { background-color: #ff8c1a; color: #0d1117; border-color: #ff8c1a; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)




# ---------------------------------------------------------------------------
# Shared thread-safe state (video thread <-> Streamlit script thread)
# ---------------------------------------------------------------------------
class SharedState:
   def __init__(self):
       self.lock = threading.Lock()
       self.gesture = "None"
       self.emotion = "Neutral"
       self.counter = 0
       self.last_sample = 0.0
       self.prev_small = None
       self.busy = False
       self.started = 0
       self.speaking = False
       self.warm_started = False
       self.prev_gesture = "None"
       self.last_spoken_at = 0.0
       self.candidate_gesture = "None"
       self.candidate_count = 0
       self.candidate_since = 0.0
       self.audio_cache = {}
       self.error = ""
       self.voice_error = ""
       self.voice_id = ""
       self.gemini_model = ""
       self.models = []
       self.models_ok = False
       self.bad_until = {}
       self.api_key = ""
       self.frames = 0
       self.calls = 0
       self.last_raw = ""


   def try_begin(self, now, small):
       """Decide whether to send this frame to Gemini. Skips static scenes to save quota."""
       with self.lock:
           if self.busy and (now - self.last_sample) > BUSY_TIMEOUT_SECONDS:
               self.busy = False
               self.error = "Gemini request timed out. Check the connection and try again."
           if self.busy or (now - self.last_sample) < SAMPLE_INTERVAL_SECONDS:
               return False
           moved = True
           if self.prev_small is not None and self.prev_small.shape == small.shape:
               moved = float(np.abs(small - self.prev_small).mean()) > MOTION_THRESHOLD
           if not moved and (now - self.last_sample) < HEARTBEAT_SECONDS:
               return False
           self.prev_small = small
           self.busy = True
           self.started += 1
           self.last_sample = now
           return True


   def finish(self, gesture, emotion, error, raw):
       with self.lock:
           self.calls += 1
           if raw:
               self.last_raw = raw
           if error:
               self.error = error
           else:
               self.gesture = gesture
               self.emotion = emotion
               self.counter += 1
               self.error = ""
           self.busy = False


   def snapshot(self):
       with self.lock:
           now = time.time()
           benched = [name for name, until in self.bad_until.items() if until > now]
           return {
               "gesture": self.gesture,
               "emotion": self.emotion,
               "counter": self.counter,
               "error": self.error,
               "voice_error": self.voice_error,
               "model": self.gemini_model,
               "frames": self.frames,
               "calls": self.calls,
               "started": self.started,
               "raw": self.last_raw,
               "benched": benched,
           }




@st.cache_resource
def get_shared_v8():
   return SharedState()




# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------
def clean_key(text):
   """Pull a usable API key out of key.txt, even if it was saved as RTF,
   wrapped in quotes, or written like GEMINI_API_KEY=xxxx."""
   text = text.strip()
   if text.startswith("{\\rtf"):
       text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
       text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
       text = re.sub(r"[{}\\;]", " ", text)
   match = re.search(r"AIza[0-9A-Za-z_\-]{35}", text)
   if match:
       return match.group(0)
   tokens = re.findall(r"[A-Za-z0-9_\-\.]{20,}", text)
   if tokens:
       return tokens[-1]
   lines = text.splitlines()
   if lines:
       return lines[0].strip().strip('"').strip("'")
   return ""





def load_gemini_key():
    # First, try Streamlit Secrets (for the deployed app)
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return clean_key(key)
    except Exception:
        pass

    # Next, try the local key.txt file
    try:
        if KEY_FILE.exists():
            text = KEY_FILE.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_key(text)
            if cleaned:
                return cleaned
    except OSError:
        pass

    # Finally, try the environment variable
    return os.environ.get("GEMINI_API_KEY", "").strip()




CONFIG_CHOICE = {}
CLIENT_CACHE = {}




def get_gemini_client(api_key):
   """Reuse one client so the connection stays open (saves a TLS handshake per call)."""
   client = CLIENT_CACHE.get(api_key)
   if client is None:
       try:
           client = genai_new.Client(api_key=api_key, http_options={"timeout": 20000})
       except Exception:  # noqa: BLE001
           client = genai_new.Client(api_key=api_key)
       CLIENT_CACHE[api_key] = client
   return client




def image_content(pil_img):
   """Send a small JPEG instead of a big PNG: much faster upload."""
   try:
       buffer = io.BytesIO()
       pil_img.convert("RGB").save(buffer, format="JPEG", quality=70)
       return genai_new.types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")
   except Exception:  # noqa: BLE001
       return pil_img




def call_gemini(api_key, model_name, pil_img):
   if GENAI_MODE == "new":
       client = get_gemini_client(api_key)
       picture = image_content(pil_img)
       # Fastest first: JSON mode, "thinking" switched off, tiny answer. If a model
       # rejects that, fall back to simpler settings (remembered per model).
       variants = [
           {
               "response_mime_type": "application/json",
               "thinking_config": {"thinking_budget": 0},
               "max_output_tokens": 100,
           },
           {"response_mime_type": "application/json"},
           None,
       ]
       start_index = CONFIG_CHOICE.get(model_name, 0)
       last_exc = None
       for index in range(start_index, len(variants)):
           config = variants[index]
           try:
               if config is None:
                   response = client.models.generate_content(
                       model=model_name, contents=[GEMINI_PROMPT, picture]
                   )
               else:
                   response = client.models.generate_content(
                       model=model_name, contents=[GEMINI_PROMPT, picture], config=config
                   )
               output = response.text or ""
               if not output.strip() and index < len(variants) - 1:
                   raise ValueError("empty reply")
               CONFIG_CHOICE[model_name] = index
               return output
           except Exception as exc:  # noqa: BLE001
               last_exc = exc
               text = str(exc)
               low = text.lower()
               if is_quota_error(text) or "404" in text:
                   raise
               config_problem = (
                   ("400" in low) or ("invalid" in low) or ("thinking" in low) or ("empty reply" in low)
               )
               if config_problem and index < len(variants) - 1:
                   CONFIG_CHOICE[model_name] = index + 1
                   continue
               raise
       raise last_exc
   if GENAI_MODE == "legacy":
       genai_old.configure(api_key=api_key)
       model = genai_old.GenerativeModel(model_name)
       response = model.generate_content([GEMINI_PROMPT, pil_img])
       return response.text or ""
   raise RuntimeError("No Gemini SDK installed (pip install google-genai).")




def parse_result(raw_text):
   match = re.search(r"\{.*\}", raw_text, re.DOTALL)
   if not match:
       raise ValueError("Gemini returned no JSON object.")
   data = json.loads(match.group(0))
   gesture_raw = str(data.get("gesture", "")).strip().lower()
   # Match longer labels first so, for example, HeadPain isn't reduced to Pain.
   canonical_labels = sorted(SPOKEN_PHRASES.keys(), key=len, reverse=True)
   normalized = re.sub(r"[^a-z0-9]+", "", gesture_raw)
   gesture = next((label for label in canonical_labels
                   if re.sub(r"[^a-z0-9]+", "", label.lower()) == normalized), "None")
   emotion = str(data.get("emotion", "Neutral")).strip().title() or "Neutral"
   return gesture, emotion




def discover_models(api_key):
   """Ask Google which Flash models this key can actually use (stable + lite first)."""
   resp = requests.get(
       "https://generativelanguage.googleapis.com/v1beta/models",
       params={"key": api_key, "pageSize": 200},
       timeout=10,
   )
   resp.raise_for_status()
   ranked = []
   for entry in resp.json().get("models", []):
       name = str(entry.get("name", "")).replace("models/", "")
       methods = entry.get("supportedGenerationMethods", [])
       low = name.lower()
       if "flash" not in low or "generateContent" not in methods:
           continue
       if any(bad in low for bad in ("image", "tts", "live", "audio", "embedding", "native", "robotics")):
           continue
       numbers = re.findall(r"\d+(?:\.\d+)?", name)
       version = float(numbers[0]) if numbers else 0.0
       is_lite = 1 if "lite" in low else 0
       is_preview = 1 if ("preview" in low or "exp" in low) else 0
       ranked.append((is_preview, 0 if is_lite else 1, -version, name))
   ranked.sort()
   return [item[3] for item in ranked]




def refresh_models(shared):
   """Build the list of models to try. Re-asks Google until it succeeds once."""
   if shared.models_ok and shared.models:
       return
   found = []
   try:
       found = discover_models(shared.api_key)
       shared.models_ok = bool(found)
   except Exception:  # noqa: BLE001
       found = []
       shared.models_ok = False
   for fallback_name in GEMINI_MODELS:
       if fallback_name not in found:
           found.append(fallback_name)
   shared.models = found




def parse_retry_seconds(message):
   """Read 'Please retry in 21h53m30.7s' style hints from a quota error."""
   match = re.search(
       r"retry in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", message, re.IGNORECASE
   )
   if not match:
       return 45.0
   hours = float(match.group(1) or 0)
   minutes = float(match.group(2) or 0)
   seconds = float(match.group(3) or 0)
   total = hours * 3600 + minutes * 60 + seconds
   if total <= 0:
       return 45.0
   return max(20.0, min(total, 86400.0))




def is_quota_error(message):
   low = message.lower()
   return ("429" in low) or ("resource_exhausted" in low) or ("quota" in low) or ("rate limit" in low)




def is_skippable_model_error(message):
   low = message.lower()
   markers = (
       "404", "not found", "not supported", "no longer available",
       "429", "resource_exhausted", "quota", "rate limit",
       "503", "unavailable", "overloaded",
   )
   return any(marker in low for marker in markers)




def maybe_speak(shared, gesture, emotion):
   """Speak right from the analysis thread, so there is no extra waiting."""
   now = time.time()
   if gesture in SPOKEN_PHRASES:
       changed = shared.prev_gesture != gesture
       stale = (now - shared.last_spoken_at) > REPEAT_SPEAK_SECONDS
       if changed or stale:
           if speak_async(shared, gesture, emotion):
               shared.last_spoken_at = now
               shared.prev_gesture = gesture
       else:
           shared.prev_gesture = gesture
   else:
       shared.prev_gesture = gesture




def analyze_frame(shared, bgr_frame):
   gesture, emotion, error, raw_text = "None", "Neutral", "", ""
   try:
       rgb = bgr_frame[:, :, ::-1]
       pil_img = Image.fromarray(np.ascontiguousarray(rgb))
       pil_img.thumbnail((384, 384))  # smaller upload can reduce network/model latency
       if not shared.api_key:
           raise RuntimeError("No Gemini key found in key.txt")


       refresh_models(shared)
       now = time.time()
       usable = [m for m in shared.models if shared.bad_until.get(m, 0) <= now]
       if not usable:
           raise RuntimeError("All Gemini models are out of quota. Put a key from a NEW Google AI Studio project (or enable billing) in key.txt, then reload.")


       reply = None
       last_exc = None
       for model_name in usable[:5]:
           try:
               reply = call_gemini(shared.api_key, model_name, pil_img)
               shared.gemini_model = model_name
               break
           except Exception as exc:  # noqa: BLE001
               last_exc = exc
               message = str(exc)
               if is_quota_error(message):
                   # Avoid trying several models when this project is already quota-limited.
                   wait = parse_retry_seconds(message)
                   until = time.time() + wait
                   for candidate in shared.models:
                       shared.bad_until[candidate] = until
                   raise RuntimeError(
                       "Gemini quota/rate limit reached. EchoSign has paused Gemini calls "
                       "until the retry window passes. Check AI Studio quota/billing, or use "
                       "a different project API key."
                   ) from exc
               if is_skippable_model_error(message):
                   shared.bad_until[model_name] = time.time() + 3600
                   continue
               raise
       if reply is None:
           raise RuntimeError("No Gemini model worked. Last: " + str(last_exc))


       raw_text = reply
       gesture, emotion = parse_result(reply)

       # Every sign, including Hello and Help, must be seen consistently to avoid
       # random labels when the user is not signing.
       now = time.time()
       with shared.lock:
           if gesture == "None" or gesture not in SPOKEN_PHRASES:
               if now - shared.candidate_since > CONFIRMATION_WINDOW_SECONDS:
                   shared.candidate_gesture = "None"
                   shared.candidate_count = 0
                   shared.candidate_since = 0.0
               confirmed_gesture = "None"
           else:
               if (gesture != shared.candidate_gesture or
                       now - shared.candidate_since > CONFIRMATION_WINDOW_SECONDS):
                   shared.candidate_gesture = gesture
                   shared.candidate_count = 1
                   shared.candidate_since = now
               else:
                   shared.candidate_count += 1
               confirmed_gesture = gesture if shared.candidate_count >= CONFIRMATION_COUNT else "None"

       # Display/speak only a confirmed label; never speak a one-frame guess.
       gesture = confirmed_gesture
       try:
           maybe_speak(shared, gesture, emotion)
       except Exception:  # noqa: BLE001
           pass
   except Exception as exc:  # noqa: BLE001
       error = "Gemini error: " + str(exc)[:200]
   finally:
       shared.finish(gesture, emotion, error, raw_text)




def make_video_callback(shared):
   def video_frame_callback(frame):
       img = frame.to_ndarray(format="bgr24")
       shared.frames += 1
       small = img[::16, ::16].mean(axis=2)
       if shared.try_begin(time.time(), small):
           snapshot = img.copy()
           threading.Thread(
               target=analyze_frame, args=(shared, snapshot), daemon=True
           ).start()
       return av.VideoFrame.from_ndarray(img, format="bgr24")


   return video_frame_callback




# ---------------------------------------------------------------------------
# ElevenLabs voice output (runs in its own thread so video never freezes)
# ---------------------------------------------------------------------------
def play_bytes(audio_bytes):
   """Play MP3 bytes. On a Mac, afplay is the fastest and needs no extra install."""
   tmp_path = None
   played = False
   try:
       if sys.platform == "darwin":
           with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
               tmp.write(audio_bytes)
               tmp_path = tmp.name
           result = subprocess.run(["afplay", tmp_path], check=False)
           played = result.returncode == 0
   except Exception:  # noqa: BLE001
       played = False
   finally:
       if tmp_path:
           try:
               os.remove(tmp_path)
           except OSError:
               pass
   if not played:
       try:
           eleven_play(audio_bytes)
       except Exception:  # noqa: BLE001
           pass




def eleven_sdk_audio(text):
   """Primary path: the exact call from the spec (voice Rachel)."""
   client = ElevenLabs(api_key=ELEVEN_API_KEY)
   try:
       audio = client.generate(
           text=text, voice="Rachel", model="eleven_multilingual_v2"
       )
   except AttributeError:
       audio = client.text_to_speech.convert(
           voice_id="21m00Tcm4TlvDq8ikWAM",
           text=text,
           model_id="eleven_multilingual_v2",
           output_format="mp3_44100_128",
       )
   if isinstance(audio, (bytes, bytearray)):
       return bytes(audio)
   return b"".join(audio)




def eleven_rest_audio(shared, text):
   """Fallback path: free ElevenLabs accounts cannot use library voices
   (like Rachel) over the API, so find a voice this account CAN use."""
   headers = {"xi-api-key": ELEVEN_API_KEY}
   voice_ids = []
   if shared.voice_id:
       voice_ids.append(shared.voice_id)
   else:
       listing = requests.get(ELEVEN_BASE + "/voices", headers=headers, timeout=20)
       listing.raise_for_status()
       voices = listing.json().get("voices", [])
       premade = [v for v in voices if v.get("category") == "premade"]
       others = [v for v in voices if v.get("category") != "premade"]
       for voice in premade + others:
           if voice.get("voice_id"):
               voice_ids.append(voice["voice_id"])
       voice_ids = voice_ids[:8]


   last_problem = "no voices found on this ElevenLabs account"
   for voice_id in voice_ids:
       resp = requests.post(
           ELEVEN_BASE + "/text-to-speech/" + voice_id,
           headers=headers,
           json={"text": text, "model_id": "eleven_multilingual_v2"},
           timeout=40,
       )
       if resp.status_code == 200 and resp.content:
           shared.voice_id = voice_id
           return resp.content
       last_problem = "HTTP " + str(resp.status_code) + " " + resp.text[:120]
   shared.voice_id = ""
   raise RuntimeError(last_problem)




def generate_audio(shared, text):
   """Get MP3 bytes for a phrase from ElevenLabs (SDK first, REST fallback)."""
   audio_bytes = None
   if not shared.voice_id:
       try:
           audio_bytes = eleven_sdk_audio(text)
       except Exception:  # noqa: BLE001
           audio_bytes = None
   if audio_bytes is None:
       audio_bytes = eleven_rest_audio(shared, text)
   return audio_bytes




def warm_voice_cache(shared):
   """Pre-generate the fixed phrases once so speaking later is instant."""
   for text in SPOKEN_PHRASES.values():
       if text in shared.audio_cache:
           continue
       try:
           shared.audio_cache[text] = generate_audio(shared, text)
       except Exception:  # noqa: BLE001
           pass




def speak_worker(shared, gesture, emotion):
   try:
       text = SPOKEN_PHRASES.get(gesture, gesture)
       audio_bytes = shared.audio_cache.get(text)
       if audio_bytes is None:
           audio_bytes = generate_audio(shared, text)
           shared.audio_cache[text] = audio_bytes
       with shared.lock:
           shared.voice_error = ""
       play_bytes(audio_bytes)
   except Exception as exc:  # noqa: BLE001
       with shared.lock:
           shared.voice_error = "ElevenLabs error: " + str(exc)[:200]
   finally:
       with shared.lock:
           shared.speaking = False




def speak_async(shared, gesture, emotion):
   """Start speaking in a background thread. Returns True if speech actually started."""
   with shared.lock:
       if shared.speaking or not ELEVEN_OK:
           return False
       shared.speaking = True
   threading.Thread(
       target=speak_worker, args=(shared, gesture, emotion), daemon=True
   ).start()
   return True




# ---------------------------------------------------------------------------
# Speech-to-sign helpers
# ---------------------------------------------------------------------------
class MicSilentError(Exception):
   """Raised when the microphone only delivers silence."""




def find_sign_sequence(sentence):
   """Every word in the sentence that has a sign, in the order it was spoken.
   Immediate repeats of the same sign are collapsed (hello hello -> hello)."""
   lowered = sentence.lower()
   hits = []
   for keywords, asset in SIGN_DICTIONARY:
       for word in keywords:
           for found in re.finditer(r"\b" + re.escape(word), lowered):
               hits.append((found.start(), word, asset))
   hits.sort()
   sequence = []
   last_asset = None
   for _, word, asset in hits:
       if asset == last_asset:
           continue
       sequence.append((word, asset))
       last_asset = asset
   return sequence




def asset_folders():
   folders = []
   for base in (BASE_DIR, Path.cwd()):
       folder = base / "assets"
       if folder.is_dir() and folder not in folders:
           folders.append(folder)
   return folders




def resolve_asset_path(relative_path):
   """Find a sign clip. Tries the exact path, then any file in assets/ with the
   same name (ignoring upper/lower case), then the same name with any video/gif
   extension, then any file whose name starts with it."""
   for base in (BASE_DIR, Path.cwd()):
       exact = base / relative_path
       if exact.is_file():
           return exact
   wanted = Path(relative_path)
   wanted_name = wanted.name.lower()
   wanted_stem = wanted.stem.lower()
   for folder in asset_folders():
       try:
           files = [f for f in folder.rglob("*") if f.is_file() and not f.name.startswith(".")]
       except OSError:
           continue
       for f in files:
           if f.name.lower() == wanted_name:
               return f
       for f in files:
           if f.stem.lower() == wanted_stem and f.suffix.lower() in ASSET_EXTENSIONS:
               return f
       for f in files:
           if f.name.lower().startswith(wanted_stem) and f.suffix.lower() in ASSET_EXTENSIONS:
               return f
   return None




def describe_assets():
   parts = []
   for base in (BASE_DIR, Path.cwd()):
       folder = base / "assets"
       if folder.is_dir():
           names = sorted(f.name for f in folder.iterdir() if not f.name.startswith("."))
           parts.append(str(folder) + " contains: " + (", ".join(names) if names else "(empty)"))
       else:
           parts.append(str(folder) + " does not exist")
   return " | ".join(parts)




def render_sign_asset(slot, relative_path):
   path = resolve_asset_path(relative_path)
   with slot.container():
       if path is None:
           st.warning("Sign asset not found: " + relative_path)
           st.caption(describe_assets())
           return
       suffix = path.suffix.lower()
       if suffix in (".mp4", ".mov", ".m4v", ".webm"):
           mime = {".webm": "video/webm", ".mov": "video/quicktime"}.get(suffix, "video/mp4")
           data = path.read_bytes()
           try:
               st.video(data, format=mime, loop=False, autoplay=True, muted=True)
           except TypeError:
               st.video(data, format=mime)
       else:
           try:
               st.image(str(path), use_container_width=True)
           except TypeError:
               st.image(str(path))




def clip_seconds(path):
   """How long a sign clip lasts, so each one can be shown exactly once."""
   seconds = 0.0
   try:
       if path.suffix.lower() == ".gif":
           with Image.open(path) as gif:
               total_ms = 0
               for index in range(getattr(gif, "n_frames", 1)):
                   gif.seek(index)
                   total_ms += int(gif.info.get("duration", 100) or 100)
           seconds = total_ms / 1000.0
       else:
           with av.open(str(path)) as container:
               if container.duration:
                   seconds = container.duration / 1000000.0
   except Exception:  # noqa: BLE001
       seconds = 0.0
   if seconds <= 0:
       seconds = 2.5
   return min(max(seconds, 0.8), 8.0)




def play_sign_sequence(sign_slot, caption_slot, signs):
   """Play each sign once, one after another, then stop."""
   total = len(signs)
   for number, (word, asset) in enumerate(signs, start=1):
       caption_slot.caption("🤟 Signing \u201c" + word + "\u201d (" + str(number) + " of " + str(total) + ")")
       sign_slot.empty()
       render_sign_asset(sign_slot, asset)
       path = resolve_asset_path(asset)
       wait = (clip_seconds(path) + 0.4) if path is not None else 1.5
       time.sleep(wait)
   sign_slot.empty()
   caption_slot.empty()




def build_audio(samples, rate):
   """Boost quiet recordings, then wrap them for the speech recognizer.
   Returns (AudioData, original_peak_level)."""
   samples = np.asarray(samples).astype(np.int32).reshape(-1)
   peak = int(np.abs(samples).max()) if samples.size else 0
   if peak < MIC_SILENT_LEVEL:
       raise MicSilentError(str(peak))
   if peak < 8000:
       gain = min(25.0, 20000.0 / peak)
       samples = np.clip(samples.astype(np.float32) * gain, -32768, 32767)
   samples = samples.astype(np.int16)
   return sr.AudioData(samples.tobytes(), int(rate), 2), peak




def record_audio(seconds):
   """Record from the default mic. Prefers sounddevice (most reliable on Mac),
   falls back to PyAudio. Returns (AudioData, peak_level)."""
   if SD_OK:
       try:
           try:
               info = sd.query_devices(kind="input")
               rate = int(info["default_samplerate"])
           except Exception:  # noqa: BLE001
               rate = 16000
           recording = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="int16")
           sd.wait()
           return build_audio(recording[:, 0], rate)
       except MicSilentError:
           raise
       except Exception:  # noqa: BLE001
           if not PYAUDIO_OK:
               raise
   recognizer = sr.Recognizer()
   with sr.Microphone() as source:
       raw = recognizer.record(source, duration=seconds)
   pcm = np.frombuffer(raw.get_raw_data(), dtype=np.int16)
   return build_audio(pcm, raw.sample_rate)




def listen_for_speech(seconds=4):
   """Returns (text, peak_level). Raises sr.UnknownValueError with the level attached."""
   audio, peak = record_audio(seconds)
   recognizer = sr.Recognizer()
   try:
       text = recognizer.recognize_google(audio, language="en-US")
   except sr.UnknownValueError as exc:
       exc.peak = peak
       raise
   return text, peak




def apply_phrase(text):
   """Store a sentence and queue the signs it contains to play once, in order."""
   signs = find_sign_sequence(text)
   st.session_state.transcript = text
   st.session_state.signs = signs
   st.session_state.play_pending = bool(signs)
   st.session_state.mic_message = ""




def set_mic_error(message):
   st.session_state.transcript = ""
   st.session_state.signs = []
   st.session_state.play_pending = False
   st.session_state.mic_message = message




def handle_typed_phrase():
   typed = str(st.session_state.get("typed_phrase", "")).strip()
   if typed:
       apply_phrase(typed)




# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
   "gesture": "None",
   "emotion": "Neutral",
   "transcript": "",
   "signs": [],
   "play_pending": False,
   "mic_message": "",
}
for state_key, state_value in DEFAULTS.items():
   if state_key not in st.session_state:
       st.session_state[state_key] = state_value


shared = get_shared_v8()
new_key = load_gemini_key()
if new_key != shared.api_key:
   # new or changed key: forget old quota pauses and re-discover models
   shared.api_key = new_key
   shared.bad_until = {}
   shared.models = []
   shared.models_ok = False
   shared.gemini_model = ""


if ELEVEN_OK and not shared.warm_started:
   shared.warm_started = True
   threading.Thread(target=warm_voice_cache, args=(shared,), daemon=True).start()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="echo-title" style="font-size:clamp(5.5rem,22vw,15rem)!important;font-weight:900!important;line-height:1.08!important;text-align:center!important;width:100%!important;background:linear-gradient(90deg,#79c0ff 0%,#d2a8ff 52%,#ffa198 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .15rem 0!important;padding:.05em 0 .12em!important;">EchoSign</p>', unsafe_allow_html=True)
st.markdown(
   '<p class="echo-subtitle">Two-way emergency medical translation bridge: '
   "sign to speech, speech to sign.</p>",
   unsafe_allow_html=True,
)


col1, col2 = st.columns(2, gap="large")


# ---------------------------------------------------------------------------
# LEFT COLUMN: Patient scanner (sign -> speech)
# ---------------------------------------------------------------------------
with col1:
   st.markdown('<div class="card-title">🧑‍⚕️ Patient Scanner</div>', unsafe_allow_html=True)
   st.markdown(
       '<div class="card-caption">Live camera. Keep both hands visible and sign clearly; detection may take a few seconds.</div>',
       unsafe_allow_html=True,
   )


   # Keep setup guidance calm and concise for a live demo.
   if GENAI_MODE is None:
       st.info("Camera sign detection needs its Gemini package installed before use.")
   elif not shared.api_key:
       st.info("Camera sign detection is not configured yet. Add your Gemini key to key.txt.")
   if not ELEVEN_OK:
       st.info("Voice output needs the ElevenLabs package installed before use.")
   elif not ELEVEN_API_KEY:
       st.caption("Voice output is not configured. Add ELEVENLABS_API_KEY to your environment to enable it.")


   ctx = webrtc_streamer(
       key="echosign-scanner",
       mode=WebRtcMode.SENDRECV,
       video_frame_callback=make_video_callback(shared),
       media_stream_constraints={
           "video": {
               "width": {"ideal": 640},
               "height": {"ideal": 480},
               "frameRate": {"ideal": 30, "max": 30},
           },
           "audio": False,
       },
       rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
       async_processing=True,
   )


   # Keep the UI focused on translation; omit the unreliable gesture/emotion cards.
   status_slot = st.empty()
   with st.expander("Developer diagnostics", expanded=False):
       st.caption("Optional troubleshooting tools. Keep this section collapsed during the demo.")
       test_gemini_clicked = st.button("🧪 Test Gemini connection")
       test_result_slot = st.empty()
       debug_slot = st.empty()





def render_debug(snap):
   raw_short = snap["raw"].replace("\n", " ")[:220] if snap["raw"] else "(none yet)"
   benched = ", ".join(snap["benched"]) if snap["benched"] else "(none)"
   debug_text = (
       "frames received: " + str(snap["frames"])
       + " | frames sent to Gemini: " + str(snap.get("started", 0))
       + " | Gemini answers: " + str(snap["calls"])
       + "\nGemini model in use: " + (snap["model"] if snap["model"] else "(finding one...)")
       + "\nmodels paused (rate limit / missing): " + benched
       + "\nlast Gemini reply: " + raw_short
       + "\nlast Gemini error: " + (snap["error"] if snap["error"] else "(none)")
       + "\nlast voice error: " + (snap["voice_error"] if snap["voice_error"] else "(none)")
   )
   debug_slot.code(debug_text, language=None)




# ---------------------------------------------------------------------------
# RIGHT COLUMN: Operator side (speech -> sign)
# NOTE: built before the live loop below so the button stays responsive.
# ---------------------------------------------------------------------------
with col2:
   st.markdown('<div class="card-title">🎙️ Operator Console</div>', unsafe_allow_html=True)
   st.markdown(
       '<div class="card-caption">Speak to the patient. Keywords are converted to sign language clips.</div>',
       unsafe_allow_html=True,
   )


   listen_clicked = st.button("🎙️ Open Attendant Microphone Listener")
   st.text_input(
       "Or type a phrase (backup if the room is too loud)",
       key="typed_phrase",
       placeholder="e.g. the doctor is coming, please wait",
       on_change=handle_typed_phrase,
   )
   replay_clicked = st.button("🔁 Replay signs")
   speech_slot = st.empty()
   caption_slot = st.empty()
   sign_slot = st.empty()


   if replay_clicked and st.session_state.signs:
       st.session_state.play_pending = True


   if listen_clicked:
       if not SR_OK:
           set_mic_error("SpeechRecognition is not installed. Run: python3 -m pip install SpeechRecognition sounddevice")
       elif not (PYAUDIO_OK or SD_OK):
           set_mic_error("No microphone library found. Run: python3 -m pip install sounddevice")
       else:
           speech_slot.info("🎧 Listening for " + str(MIC_SECONDS) + " seconds... speak NOW.")
           time.sleep(0.4)
           try:
               heard, level = listen_for_speech(MIC_SECONDS)
               apply_phrase(heard)
           except MicSilentError as exc:
               set_mic_error(
                   "Microphone is only delivering silence (level " + str(exc) + "). "
                   "On Mac: System Settings > Privacy & Security > Microphone, turn ON your "
                   "Terminal / VS Code / PyCharm, then fully quit and reopen it. "
                   "Also check System Settings > Sound > Input is the mic you are speaking into."
               )
           except sr.UnknownValueError as exc:
               level = getattr(exc, "peak", "?")
               set_mic_error(
                   "Heard sound (level " + str(level) + ") but could not make out words. "
                   "Speak louder and closer, start right after the Listening message, "
                   "and say clear keywords like 'help', 'doctor' or 'nurse'."
               )
           except sr.RequestError as exc:
               set_mic_error("Speech service error (check your internet): " + str(exc))
           except Exception as exc:  # noqa: BLE001
               set_mic_error("Microphone error: " + str(exc))


   if st.session_state.mic_message:
       speech_slot.warning(st.session_state.mic_message)
   elif st.session_state.transcript:
       speech_slot.success("🗣️ " + st.session_state.transcript)


   signs = st.session_state.signs
   if signs and st.session_state.play_pending:
       st.session_state.play_pending = False
       play_sign_sequence(sign_slot, caption_slot, signs)
   elif signs:
       caption_slot.empty()
   elif st.session_state.transcript:
       sign_slot.info("No matching medical sign found in that sentence.")


if test_gemini_clicked:
   with st.spinner("Contacting Gemini..."):
       started_at = time.time()
       lines = []
       try:
           if not shared.api_key:
               raise RuntimeError("No Gemini key found in key.txt")
           visible = discover_models(shared.api_key)
           lines.append("models this key can see: " + (", ".join(visible[:8]) if visible else "(none)"))
           blank = Image.new("RGB", (320, 240), (40, 40, 40))
           worked = False
           for model_name in (visible[:6] if visible else GEMINI_MODELS):
               t0 = time.time()
               try:
                   reply_text = call_gemini(shared.api_key, model_name, blank)
                   lines.append(
                       "OK     " + model_name + " in " + str(round(time.time() - t0, 1)) + "s  reply: "
                       + reply_text.replace("\n", " ")[:70]
                   )
                   shared.gemini_model = model_name
                   shared.bad_until.pop(model_name, None)
                   worked = True
                   break
               except Exception as exc:  # noqa: BLE001
                   message = str(exc)
                   if is_quota_error(message):
                       shared.bad_until[model_name] = time.time() + parse_retry_seconds(message)
                       lines.append("QUOTA  " + model_name + " is out of free requests")
                   else:
                       lines.append("FAIL   " + model_name + ": " + message[:120])
           if not worked:
               lines.append("No model worked. If quota-limited, wait for reset or check Google AI Studio billing/limits.")
       except Exception as exc:  # noqa: BLE001
           lines.append("Gemini test FAILED: " + str(exc)[:300])
       test_result_slot.code("\n".join(lines), language=None)


# ---------------------------------------------------------------------------
# LIVE STATUS: take one snapshot per normal Streamlit run.
# The old while-loop repeatedly redrew the entire page while the camera was live,
# interrupting speech-to-sign translation and making the UI appear to refresh.
snap = shared.snapshot()
st.session_state.gesture = snap["gesture"]
st.session_state.emotion = snap["emotion"]
render_debug(snap)

if ctx.state.playing or ctx.state.signalling:
   if snap["error"]:
       status_slot.caption("🟠 Sign detection is temporarily unavailable. Open Developer diagnostics for details.")
   elif snap["voice_error"]:
       status_slot.caption("🟠 Voice output needs attention. Open Developer diagnostics for details.")
   else:
       status_slot.caption("🟢 Camera connected. Sign detection updates as frames are analyzed.")
else:
   status_slot.caption("⚪ Camera is off. Press START above to begin.")
