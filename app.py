"""
Doodle Guessing Game  -  Gradio app

Project layout (everything sits next to this file):
    app.py
    characters/   Happy.jpeg  Thinking.jpeg  Smile.jpeg  Celebrate.jpeg  confused.jpeg
    models/       DOODLE_MODEL_1.pth  class_names.json

Run:   pip install gradio torch pillow numpy
       python app.py
"""
import base64
import csv
import inspect
import io
import json
import re
import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch import nn

# ------------------------------------------------------------------ settings
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "DOODLE_MODEL_1.pth"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"
CHAR_DIR = BASE_DIR / "characters"
FEEDBACK_LOG = BASE_DIR / "feedback_log.csv"   # every Yes/No press is saved here
LOG_FEEDBACK = True

HIDDEN_UNITS = 32        # must match the model you trained
THINK_SECONDS = 0.6      # minimum time the "Thinking" pose stays visible
PAD_RATIO = 0.02         # empty border around the doodle; Quick Draw doodles nearly fill the 28x28 frame
TARGET_STROKE = 0.06     # wanted line thickness as a fraction of the frame (Quick Draw is ~1.5-2 px of 28)

# ------------------------------------------------------------------ model
class TinyVGG2(nn.Module):
    """Same architecture as the notebook (conv_block_1, conv_block_2, classifier)."""

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape),   # 32*7*7 = 1568
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


with open(CLASSES_PATH, encoding="utf-8") as f:
    _raw = json.load(f)
if isinstance(_raw, dict):                         # tolerate {"0": "airplane"} or {"airplane": 0}
    try:
        _raw = [_raw[k] for k in sorted(_raw, key=int)]
    except ValueError:
        _raw = sorted(_raw, key=_raw.get)
CLASS_NAMES = list(_raw)

model = TinyVGG2(1, HIDDEN_UNITS, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


@torch.inference_mode()
def predict(x):
    """x: tensor (1, 1, 28, 28) -> top-3 list of (class_name, probability)."""
    probs = torch.softmax(model(x), dim=1)[0]
    p, i = probs.topk(3)
    return [(CLASS_NAMES[j], float(q)) for q, j in zip(p, i)]


# ------------------------------------------------------------------ image preprocessing
def to_pil(value):
    """Gradio hands over PIL / numpy / an ImageEditor dict depending on the component."""
    if value is None:
        return None
    if isinstance(value, dict):                    # ImageEditor -> {"background", "layers", "composite"}
        parts = [value.get("composite")] + list(value.get("layers") or []) + [value.get("background")]
        value = next((p for p in parts if p is not None), None)
        if value is None:
            return None
    if isinstance(value, np.ndarray):
        value = Image.fromarray(value.astype("uint8"))
    return value


def flatten(img, max_side=640):
    """Any image -> RGB on a white background (for display)."""
    rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.alpha_composite(rgba)
    out = bg.convert("RGB")
    out.thumbnail((max_side, max_side))
    return out


def preprocess(img):
    """
    Turn a canvas drawing / uploaded picture into what the CNN saw in training:
    28x28, white strokes on a black background, doodle centred and filling the frame.
    Returns (tensor [1,1,28,28], 28x28 PIL preview) or (None, None) if nothing is drawn.
    """
    gray = np.asarray(flatten(img, max_side=1000).convert("L"), dtype=np.float32) / 255.0

    # 1. make strokes bright and background dark (canvas = dark on light, Quick Draw = light on dark).
    #    Doodles are sparse, so the MEDIAN pixel is the background (the border is not reliable:
    #    a doodle that touches the frame edges would fool it).
    ink = 1.0 - gray if np.median(gray) > 0.5 else gray.copy()
    bg = float(np.median(ink))
    if ink.max() - bg < 0.2:                       # blank canvas
        return None, None
    ink = np.clip((ink - bg) / (ink.max() - bg), 0, 1)
    ink[ink < 0.2] = 0

    # 2. crop to the doodle, then fit it (keeping its shape) into a centred 112x112 frame.
    #    112 = 4 x 28, so line thickness can be matched cheaply before shrinking to 28x28.
    ys, xs = np.where(ink > 0.3)
    if len(xs) < 15:
        return None, None
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = 112 * (1 - 2 * PAD_RATIO) / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    piece = Image.fromarray((crop * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    frame = Image.new("L", (112, 112), 0)
    frame.paste(piece, ((112 - new_w) // 2, (112 - new_h) // 2))
    arr = np.asarray(frame, dtype=np.float32) / 255.0
    arr /= arr.max() + 1e-6

    # 3. estimate the current line width (area / outline length) and thicken thin lines
    mask = arr > 0.5
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    thickness = 2.0 * mask.sum() / max(int((mask & ~inner).sum()), 1)
    radius = int(round((TARGET_STROKE * 112 - thickness) / 2))
    if radius >= 1:
        thick = Image.fromarray((arr * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * radius + 1))
        arr = np.asarray(thick, dtype=np.float32) / 255.0

    # 4. shrink to 28x28 (averaging gives the soft edges Quick Draw has), peak brightness = 1
    small = Image.fromarray((arr * 255).astype(np.uint8)).resize((28, 28), Image.BOX)
    final = np.asarray(small, dtype=np.float32)
    final /= final.max() + 1e-6
    tensor = torch.from_numpy(final).unsqueeze(0).unsqueeze(0)           # (1, 1, 28, 28), values 0..1
    return tensor, Image.fromarray((final * 255).astype(np.uint8))


def data_uri(img, fmt="PNG", scale=1):
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ characters (case-insensitive lookup)
def load_character(stem):
    for p in CHAR_DIR.glob("*"):
        if p.stem.lower() == stem.lower() and p.suffix.lower() in {".jpeg", ".jpg", ".png", ".webp"}:
            img = flatten(Image.open(p), max_side=420)
            return data_uri(img, "JPEG")
    print(f"[warning] character image '{stem}' not found in {CHAR_DIR}")
    return None


CHARS = {
    "happy": load_character("Happy"),
    "thinking": load_character("Thinking"),
    "smile": load_character("Smile"),
    "celebrate": load_character("Celebrate"),
    "confused": load_character("confused"),
}

EMOJI = {
    "cat": "🐱", "tiger": "🐯", "lion": "🦁", "dog": "🐶", "bear": "🐻", "rabbit": "🐰",
    "horse": "🐴", "elephant": "🐘", "giraffe": "🦒", "butterfly": "🦋", "fish": "🐟",
    "snake": "🐍", "whale": "🐋", "dolphin": "🐬", "shark": "🦈", "apple": "🍎", "pear": "🍐",
    "banana": "🍌", "pizza": "🍕", "ice cream": "🍦", "donut": "🍩", "birthday cake": "🎂",
    "hamburger": "🍔", "bicycle": "🚲", "car": "🚗", "bus": "🚌", "airplane": "✈️",
    "chair": "🪑", "clock": "🕒", "umbrella": "☂️", "light bulb": "💡", "scissors": "✂️",
    "guitar": "🎸", "key": "🔑", "sun": "☀️", "moon": "🌙", "cloud": "☁️", "tree": "🌳",
    "star": "⭐", "eye": "👁️", "hand": "✋", "shoe": "👟", "hat": "🎩", "house": "🏠",
    "smiley face": "😊",
}


def pretty(name):
    return name.replace("_", " ")


def emoji(name):
    return EMOJI.get(pretty(name), "🎨")


# ------------------------------------------------------------------ HTML pieces
WELCOME = (
    "Hey! I'm <b>DoodleBot</b> 🤖<br>I'll look at your image and try to guess what it is!"
    "<br><br>Ready?<br><span style='opacity:.7'>(You can draw or upload an image)</span>"
)


def stage_html(mood, text):
    src = CHARS.get(mood)
    img = f'<img src="{src}" alt="{mood}">' if src else '<div class="no-img">🤖</div>'
    return (f'<div class="stage mood-{mood}"><div class="stage-img">{img}</div>'
            f'<div class="stage-bubble">{text}</div></div>')


def chat_html(messages, thinking=False):
    items = list(messages)
    if thinking:
        items.append(("bot", 'Hmm... let me take a look! 🤔 <span class="dots"><i></i><i></i><i></i></span>'))
    if not items:
        return '<div class="chat empty">Your chat with DoodleBot will show up here ✨</div>'
    rows = "".join(f'<div class="msg {who}"><div class="bubble">{html}</div></div>'
                   for who, html in reversed(items))          # column-reverse keeps newest at the bottom
    return f'<div class="chat">{rows}</div>'


def result_card(top3, thumb):
    name, p = top3[0]
    others = " &middot; ".join(f"{pretty(n)} {emoji(n)} {q * 100:.0f}%" for n, q in top3[1:])
    return (
        "Hmm... let me take a look!<br>This looks like a..."
        '<div class="result"><div class="result-main"><div>'
        f'<div class="result-name">{pretty(name)} {emoji(name)}</div>'
        '<div class="result-sub">Am I right?</div></div>'
        f'<img class="thumb" src="{thumb}" title="What I saw (28x28)"></div>'
        f'<div class="bar"><span style="width:{p * 100:.0f}%"></span></div>'
        f'<div class="result-conf">Confidence {p * 100:.0f}% &nbsp;|&nbsp; other guesses: {others}</div></div>'
    )


def score_line(state):
    return f'<div class="score">Score: {state["right"]} / {state["total"]} correct</div>'


# ------------------------------------------------------------------ app logic
def new_state():
    # shown = image in the "Your Image" panel, upload = last uploaded picture (kept server-side)
    return {"chat": [], "last": None, "right": 0, "total": 0, "shown": None, "upload": None}


def log_feedback(top3, correct):
    if not LOG_FEEDBACK or not top3:
        return
    try:
        is_new = not FEEDBACK_LOG.exists()
        with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["time", "prediction", "confidence", "was_correct", "top3"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), top3[0][0],
                        f"{top3[0][1]:.3f}", int(correct), "|".join(n for n, _ in top3)])
    except OSError:
        pass


USER_MSG = "Here's my doodle! 🎨"


def on_thinking(state):
    """Runs instantly on click (it does not wait for the canvas) -> Thinking pose."""
    state = state or new_state()
    return (stage_html("thinking", "Hmm... let me take a look! 🤔"),
            chat_html(state["chat"] + [("user", USER_MSG)], thinking=True),
            gr.update(visible=False))


def on_guess(sketch, state):
    """Reads the drawing (or the uploaded image), runs the CNN, shows the Smile pose + answer."""
    state = state or new_state()
    time.sleep(THINK_SECONDS)                                     # keep the Thinking pose visible for a moment

    source, x, thumb, from_drawing = None, None, None, False
    drawn = to_pil(sketch)
    if drawn is not None:
        x, thumb = preprocess(drawn)
        if x is not None:
            source, from_drawing = drawn, True
    if x is None and state["upload"] is not None:                 # nothing drawn -> use the uploaded image
        x, thumb = preprocess(state["upload"])
        source = state["upload"]

    if x is None:
        state["chat"].append(("bot", "I can't see a doodle yet 🙈<br>Draw something on the pad "
                                     "or upload an image, then press <b>Guess!</b>"))
        return (stage_html("happy", "Draw something first and I'll guess it! ✏️"),
                chat_html(state["chat"]), gr.update(visible=False), state["shown"], state)

    if from_drawing:
        state["upload"] = None                                    # the drawing is now the active input
    state["shown"] = flatten(source)
    top3 = predict(x)
    state["last"] = top3
    state["chat"] += [("user", USER_MSG), ("bot", result_card(top3, data_uri(thumb, scale=3)))]
    return (stage_html("smile", "I think I've got it! 😄<br>Tell me if I'm right 👇"),
            chat_html(state["chat"]), gr.update(visible=True), state["shown"], state)


def on_yes(state):
    state = state or new_state()
    last = state["last"]
    if not last:
        return stage_html("happy", "Draw something and press Guess! ✏️"), chat_html(state["chat"]), \
            gr.update(visible=False), state
    state["right"] += 1
    state["total"] += 1
    log_feedback(last, True)
    state["last"] = None
    state["chat"] += [("user", "Yes! 🎉"),
                      ("bot", f"Woohoo! 🎉 I knew it was a <b>{pretty(last[0][0])}</b> {emoji(last[0][0])}!"
                              f"{score_line(state)}")]
    return (stage_html("celebrate", "Yay! I knew it! 🎉<br>Want to try another one?"),
            chat_html(state["chat"]), gr.update(visible=False), state)


def on_no(state):
    state = state or new_state()
    last = state["last"]
    if not last:
        return stage_html("happy", "Draw something and press Guess! ✏️"), chat_html(state["chat"]), \
            gr.update(visible=False), state
    state["total"] += 1
    log_feedback(last, False)
    state["last"] = None
    alt = " or ".join(f"<b>{pretty(n)}</b> {emoji(n)}" for n, _ in last[1:])
    state["chat"] += [("user", "No 🙅"),
                      ("bot", f"Oh no, I got confused! 😅<br>Maybe it's a {alt}?<br>"
                              f"Try again with a few more details and I'll try harder!{score_line(state)}")]
    return (stage_html("confused", "Oops! I got confused 😅<br>Draw it again and I'll try harder!"),
            chat_html(state["chat"]), gr.update(visible=False), state)


def on_upload(path, state):
    state = state or new_state()
    if isinstance(path, list):
        path = path[0] if path else None
    path = getattr(path, "name", path)
    if path:
        img = flatten(ImageOps.exif_transpose(Image.open(path)))
        state["upload"], state["shown"] = img, img
    return state["shown"], None, state                           # show it, and wipe the drawing pad


def on_clear(state):
    state = state or new_state()
    state["last"], state["upload"], state["shown"] = None, None, None
    return (stage_html("happy", "Ready for the next one? Draw or upload something! ✏️"),
            chat_html(state["chat"]), gr.update(visible=False), None, state, None)   # last None = drawing pad


def bot_reply(text):
    t = text.lower()
    if re.search(r"\b(hi|hello|hey|yo)\b", t):
        return "Hi there! 👋 Draw something and press <b>Guess!</b>"
    if re.search(r"\b(thanks?|thx)\b", t):
        return "You're welcome! 😄"
    if re.search(r"\b(who|name)\b", t):
        return "I'm DoodleBot 🤖 a small CNN that has studied thousands of doodles!"
    if re.search(r"\b(categories|category|list|recogni[sz]e|know)\b|what can", t):
        return "I know these: " + ", ".join(f"{pretty(n)} {emoji(n)}" for n in CLASS_NAMES)
    if re.search(r"\b(help|how)\b", t):
        return ("Easy! ✏️ Draw on the pad (or upload an image) and press <b>Guess!</b>. "
                "Then tell me if I got it right.")
    return "I'm better at doodles than chatting 😅 Draw something and I'll guess it!"


def _escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def on_chat(text, state):
    state = state or new_state()
    text = (text or "").strip()
    if text:
        state["chat"] += [("user", _escape(text)), ("bot", bot_reply(text))]
    return chat_html(state["chat"]), "", state


# ------------------------------------------------------------------ styling
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600&family=Patrick+Hand&display=swap');
.gradio-container {background: linear-gradient(180deg,#f6f8ff,#eceffd) !important;
  font-family:'Fredoka',system-ui,sans-serif !important; max-width:1180px !important; width:100% !important;
  margin-left:auto !important; margin-right:auto !important;}
#header {background:#e8ecff; border:1px solid #d9e0ff; border-radius:22px; padding:20px 28px;
  display:flex; flex-wrap:wrap; align-items:center; justify-content:center; gap:10px 22px;}
#header .logo {font-size:3rem;}
#header .title {font-size:2.3rem; font-weight:600; color:#1f2a5c; line-height:1.1;}
#header .sub {color:#5b6591; margin-top:4px;}
#header .fool {font-family:'Patrick Hand',cursive; font-size:1.5rem; color:#4b5bd6; transform:rotate(-6deg);
  white-space:nowrap;}
.card {background:#fff !important; border:1px solid #e3e7fb !important; border-radius:22px !important;
  box-shadow:0 6px 24px rgba(70,90,200,.08) !important; padding:16px !important;}
.sec {font-weight:600; color:#1f2a5c; font-size:1.1rem; margin:4px 0;}

.stage {display:flex; align-items:center; gap:16px;}
.stage-img {flex:0 0 190px;}
.stage-img img {width:190px; height:auto; mix-blend-mode:multiply; border-radius:18px;}
.no-img {font-size:6rem; text-align:center;}
.stage-bubble {position:relative; flex:1; background:#e8efff; border:1px solid #d3deff; border-radius:20px;
  padding:16px 18px; color:#1f2a5c; font-size:1.1rem; line-height:1.45;}
.stage-bubble:before {content:""; position:absolute; left:-9px; top:36px; border:9px solid transparent;
  border-left:0; border-right-color:#e8efff;}
.mood-happy .stage-img img, .mood-smile .stage-img img {animation:bob 2.4s ease-in-out infinite;}
.mood-thinking .stage-img img {animation:wiggle 1s ease-in-out infinite;}
.mood-celebrate .stage-img img {animation:pop .6s ease-out, bob 1.6s ease-in-out .6s infinite;}
.mood-confused .stage-img img {animation:shake .6s ease-in-out;}
@keyframes bob {0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)}}
@keyframes wiggle {0%,100%{transform:rotate(-3deg)} 50%{transform:rotate(3deg)}}
@keyframes pop {0%{transform:scale(.8)} 60%{transform:scale(1.12)} 100%{transform:scale(1)}}
@keyframes shake {0%,100%{transform:translateX(0)} 20%{transform:translateX(-8px)} 40%{transform:translateX(8px)}
  60%{transform:translateX(-5px)} 80%{transform:translateX(5px)}}

.chat {display:flex; flex-direction:column-reverse; gap:10px; height:380px; overflow-y:auto; padding:6px 4px;}
.chat.empty {flex-direction:row; align-items:center; justify-content:center; color:#9aa2cc;}
.msg {display:flex;} .msg.user {justify-content:flex-end;}
.bubble {max-width:90%; padding:12px 15px; border-radius:18px; line-height:1.45; color:#1f2a5c;
  background:#eef2ff; border:1px solid #dfe5ff;}
.msg.user .bubble {background:#5b6cff; border-color:#5b6cff; color:#fff;}
.result {margin-top:8px; background:#fff; border:1px solid #d9e0ff; border-radius:14px; padding:12px 14px;}
.result-main {display:flex; justify-content:space-between; align-items:center; gap:10px;}
.result-name {font-size:1.9rem; font-weight:600;} .result-sub {color:#5b6591;}
.thumb {width:60px; height:60px; border-radius:8px; image-rendering:pixelated; background:#000;}
.bar {height:8px; background:#e6eaff; border-radius:99px; margin:10px 0 6px; overflow:hidden;}
.bar span {display:block; height:100%; background:linear-gradient(90deg,#5b6cff,#8aa0ff); border-radius:99px;}
.result-conf {font-size:.85rem; color:#6a74a6;} .score {font-size:.85rem; color:#6a74a6; margin-top:6px;}
.dots i {display:inline-block; width:6px; height:6px; margin:0 2px; border-radius:50%; background:#5b6cff;
  animation:blink 1s infinite;}
.dots i:nth-child(2){animation-delay:.2s} .dots i:nth-child(3){animation-delay:.4s}
@keyframes blink {0%,80%,100%{opacity:.2} 40%{opacity:1}}

#yes-btn {background:#37a96b !important; color:#fff !important; border:none !important; border-radius:14px !important;
  font-weight:600 !important; font-size:1.05rem !important;}
#no-btn {background:#eceff9 !important; color:#46507a !important; border:none !important; border-radius:14px !important;
  font-weight:600 !important; font-size:1.05rem !important;}
#guess-btn {background:linear-gradient(90deg,#5b6cff,#7b8dff) !important; color:#fff !important; border:none !important;
  border-radius:14px !important; font-weight:600 !important; font-size:1.1rem !important;}
#upload-btn {background:#6c8cff !important; color:#fff !important; border:none !important; border-radius:14px !important;
  font-weight:500 !important;}
#clear-btn {background:#eef0f8 !important; color:#46507a !important; border:none !important; border-radius:14px !important;}
#send-btn {background:#5b6cff !important; color:#fff !important; border:none !important; border-radius:14px !important;}
.chips {display:flex; flex-wrap:wrap; gap:8px;}
.chip {background:#eef2ff; border:1px solid #dfe5ff; border-radius:99px; padding:4px 12px; color:#1f2a5c; font-size:.95rem;}
#pad-mat, #preview-box {background:#eef1ff !important; border:2px dashed #7b8dff !important; border-radius:18px !important;}
#pad-mat {padding:10px !important;}
#preview-box .wrap {background:transparent !important;}
footer {display:none !important;}
@media (max-width:700px){ .stage{flex-direction:column;} .chat{height:300px;} }
"""


def supported(func, **kwargs):
    """Keep only the keyword arguments this Gradio version understands (Gradio 4/5/6 differ slightly)."""
    names = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in names}


THEME = gr.themes.Soft(primary_hue="indigo")

# The design is a light theme, so always open the page in light mode (even if Windows/Chrome is in dark mode)
FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.toString());
  }
}
"""

# ------------------------------------------------------------------ interface
with gr.Blocks(**supported(gr.Blocks.__init__, title="Doodle Guessing Game", css=CSS, theme=THEME, js=FORCE_LIGHT_JS)) as demo:
    state = gr.State(new_state())

    gr.HTML(
        '<div id="header"><div class="logo">🎨</div><div><div class="title">Doodle Guessing Game</div>'
        '<div class="sub">Draw an image or upload one, and let my character guess what it is!</div></div>'
        '<div class="fool">Can you fool me? 😉</div></div>'
    )

    with gr.Row(equal_height=False):
        # ---------------- left: DoodleBot + chat
        with gr.Column(scale=6, elem_classes="card"):
            stage = gr.HTML(stage_html("happy", WELCOME))
            chat = gr.HTML(chat_html([]))
            with gr.Row(visible=False) as fb_row:
                yes_btn = gr.Button("✅ Yes! 🎉", elem_id="yes-btn")
                no_btn = gr.Button("❌ No", elem_id="no-btn")
            with gr.Row():
                msg = gr.Textbox(placeholder="Type your message...", show_label=False, container=False,
                                 max_lines=1, scale=8)
                send_btn = gr.Button("➤", elem_id="send-btn", scale=1, min_width=56)

        # ---------------- right: image + drawing pad
        with gr.Column(scale=5, elem_classes="card"):
            gr.HTML('<div class="sec">✏️ Draw here</div>')
            with gr.Column(elem_id="pad-mat"):
                sketch = gr.ImageEditor(**supported(
                    gr.ImageEditor.__init__,
                    type="pil", sources=(), transforms=(), layers=False, show_label=False,
                    canvas_size=(400, 400), height=420,
                    placeholder="Pick the brush tool on the left, then draw here ✏️",
                    brush=gr.Brush(default_size=10, colors=["#000000"], default_color="#000000", color_mode="fixed"),
                    eraser=gr.Eraser(default_size=24),
                ))
            guess_btn = gr.Button("🔮 Guess!", elem_id="guess-btn")

            gr.HTML('<div class="sec">🖼️ Your Image</div>')
            preview = gr.Image(**supported(gr.Image.__init__, type="pil", interactive=False,
                                           show_label=False, height=280, elem_id="preview-box"))
            with gr.Row():
                clear_btn = gr.Button("🗑 Clear", elem_id="clear-btn")
                upload_btn = gr.UploadButton("⬆ Upload Image", file_types=["image"], elem_id="upload-btn")

    with gr.Accordion("🧠 What can I recognize?", open=False):
        gr.HTML('<div class="chips">' +
                "".join(f'<span class="chip">{emoji(n)} {pretty(n)}</span>' for n in CLASS_NAMES) + "</div>")

    # ---------------- wiring
    # show_progress="hidden" removes Gradio's grey loading overlay so the character stays clearly visible
    quiet = dict(show_progress="hidden")
    pose_outputs = [stage, chat, fb_row]
    result_outputs = [stage, chat, fb_row, preview, state]
    # Guess: the Thinking pose appears instantly, then the real prediction runs
    guess_btn.click(on_thinking, [state], pose_outputs, queue=False, **quiet) \
        .then(on_guess, [sketch, state], result_outputs, **quiet)
    upload_btn.upload(on_upload, [upload_btn, state], [preview, sketch, state], **quiet) \
        .then(on_thinking, [state], pose_outputs, queue=False, **quiet) \
        .then(on_guess, [sketch, state], result_outputs, **quiet)
    yes_btn.click(on_yes, [state], pose_outputs + [state], **quiet)
    no_btn.click(on_no, [state], pose_outputs + [state], **quiet)
    clear_btn.click(on_clear, [state], result_outputs + [sketch], **quiet)
    msg.submit(on_chat, [msg, state], [chat, msg, state], **quiet)
    send_btn.click(on_chat, [msg, state], [chat, msg, state], **quiet)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )