import os, random, uuid, math, textwrap, json, time
from pathlib import Path
from datetime import datetime
import requests

from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment

import numpy as np
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, VideoClip
from moviepy.video.fx.all import crop
from PIL import Image, ImageDraw, ImageFont

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# ==========================================
# === ENV & PATHS ===
# ==========================================
load_dotenv()
ROOT = Path(__file__).parent
DATA = ROOT / "data"
OVERLAYS = DATA / "overlays"
AUTOSEEDS = DATA / "autoseeds.txt"
OUT = ROOT / "out"
for p in (DATA, OVERLAYS, OUT):
    p.mkdir(exist_ok=True)

LANG = os.getenv("LANG", "en")
VIDEO_DURATION = int(os.getenv("VIDEO_DURATION", "90"))
SCENES_MIN = int(os.getenv("SCENES_MIN", "6"))
SCENES_MAX = int(os.getenv("SCENES_MAX", "9"))
MIN_AUDIO_MS = VIDEO_DURATION * 1000

# ==========================================
# === SEED SYSTEM ===
# ==========================================
BASE_SEEDS = [
    "God is not gone. He's buried.",
    "The Pulse counts backwards.",
    "We kept digging.",
    "Silence replied in numbers.",
    "No maps survived the light.",
    "Eclipsera's soil remembers footsteps that never existed.",
    "The choir is a machine misfiring a prayer.",
    "Report says light is safe; the light disagrees.",
    "If you hear the Pulse, do not count with it; it counts you.",
]

HOOKS = ["Transmission begins:", "Archive fragment:", "Field log:", "After the collapse:"]
TURNS = ["We kept digging.", "The signal grew teeth.", "Silence replied in numbers.", "No maps survived the light."]

REPLACE_MAP = {
    "God": "the divine",
    "light": "signal",
    "darkness": "static",
    "soil": "veil",
    "bones": "wires",
    "blood": "current",
    "song": "frequency",
    "prayer": "protocol",
}


def load_seeds():
    lines = []
    if AUTOSEEDS.exists():
        for s in AUTOSEEDS.read_text(encoding="utf-8").splitlines():
            s = s.strip()
            if s:
                lines.append(s)
    if not lines:
        lines = BASE_SEEDS[:]
    return list(dict.fromkeys(lines))  # unique preserve order


def mutate_line(s: str) -> str:
    words = s.split()
    if not words:
        return s
    for i, w in enumerate(words):
        key = w.strip(".,;:!?\"'")
        if key in REPLACE_MAP and random.random() < 0.25:
            words[i] = w.replace(key, REPLACE_MAP[key])
    if random.random() < 0.15 and len(words) > 4:
        j = random.randrange(1, len(words) - 1)
        words.insert(j, random.choice(["slowly", "again", "beneath"]))
    return " ".join(words)


def evolve_and_store(new_lines):
    pool = load_seeds()
    out = pool[:]
    for ln in new_lines:
        m = mutate_line(ln)
        if m not in out:
            out.append(m)
    AUTOSEEDS.write_text("\n".join(out[-500:]), encoding="utf-8")


def make_script(total_sentences=8):
    seeds = load_seeds()
    hook = random.choice(HOOKS)
    body = random.sample(seeds, k=min(total_sentences, len(seeds)))
    turn = random.choice(TURNS)
    full = " ".join([hook] + body + [turn])
    full = " ".join(full.split())
    if len(full) > 500:
        full = full[:497] + "…"
    return full, body


# ==========================================
# === AUDIO DESIGN ===
# ==========================================
def _pitch_down(seg, semitones=-2.5):
    new_fr = int(seg.frame_rate * (2.0 ** (semitones / 12.0)))
    return seg._spawn(seg.raw_data, overrides={"frame_rate": new_fr}).set_frame_rate(seg.frame_rate)


def _echo(seg, delay_ms=240, decay_db=-10):
    echo_part = seg - 6
    echo_part = AudioSegment.silent(duration=delay_ms) + (echo_part + decay_db)
    return seg.overlay(echo_part)


def _drone(duration_ms, freq=62.0, vol_db=-24):
    sr = 44100
    t = np.linspace(0, duration_ms / 1000.0, int(sr * duration_ms / 1000.0), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16).tobytes()
    return AudioSegment(data=wave, sample_width=2, frame_rate=sr, channels=1) + vol_db


def tts_and_design(text, out_mp3):
    gTTS(text=text, lang="en").save(out_mp3)
    v = AudioSegment.from_file(out_mp3).set_channels(1).set_frame_rate(44100)
    v = _pitch_down(v, -2.5)
    v = _echo(v, 240, -10)
    if len(v) < MIN_AUDIO_MS:
        v = v + AudioSegment.silent(duration=MIN_AUDIO_MS - len(v))
    d = _drone(len(v), freq=random.choice([49.0, 62.0, 73.0]), vol_db=-25)
    mix = _echo(d.overlay(v), 410, -14)
    out2 = str(OUT / f"voice_{uuid.uuid4().hex[:6]}.mp3")
    mix.export(out2, format="mp3")
    return out2


# ==========================================
# === CAPTION (PIL) ===
# ==========================================
def caption_png(text: str, width=1000, padding=30):
    wrapped = textwrap.fill(text, 30)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 56)
    except Exception:
        font = ImageFont.load_default()
    dummy = Image.new("RGBA", (width, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    bbox = d.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=6)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img = Image.new("RGBA", (width + 2 * padding, h + 2 * padding), (0, 0, 0, 0))
    sd = ImageDraw.Draw(img)
    for ox, oy in ((1, 1), (2, 2), (-1, 1), (1, -1), (-2, -2)):
        sd.multiline_text(
            (img.width // 2 + ox, padding + oy),
            wrapped,
            font=font,
            fill=(0, 0, 0, 150),
            align="center",
            anchor="ma",
            spacing=6,
        )
    d2 = ImageDraw.Draw(img)
    d2.multiline_text(
        (img.width // 2, padding),
        wrapped,
        font=font,
        fill=(255, 255, 255, 255),
        align="center",
        anchor="ma",
        spacing=6,
    )
    out = str(OUT / f"cap_{uuid.uuid4().hex[:6]}.png")
    img.save(out)
    return out


# ==========================================
# === BACKGROUND FETCH ===
# ==========================================
COMMONS_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
KEYWORDS = ["nebula", "galaxy", "desert night", "cathedral ruin", "alien landscape", "stars", "eclipse"]


def fetch_backgrounds(limit=8, timeout=12):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": "|".join(KEYWORDS),
        "gsrlimit": str(limit),
        "gsrnamespace": "6",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": "1080",
    }
    try:
        r = requests.get(COMMONS_ENDPOINT, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        urls = []
        for _, p in pages.items():
            info = p.get("imageinfo", [{}])[0]
            url = info.get("thumburl") or info.get("url")
            if url and url.lower().endswith((".jpg", ".jpeg", ".png")):
                urls.append(url)
        if not urls:
            return 0
        for f in OVERLAYS.glob("*.*"):
            f.unlink(missing_ok=True)
        ok = 0
        for u in urls:
            name = u.split("/")[-1].split("?")[0]
            path = OVERLAYS / name
            img = requests.get(u, timeout=timeout)
            img.raise_for_status()
            path.write_bytes(img.content)
            ok += 1
        return ok
    except Exception:
        return 0


# ==========================================
# === VIDEO GEN ===
# ==========================================
def procedural_frame(t, w=1080, h=1920):
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dy = (y - cy) / (h / 2)
    dx = (x - cx) / (w / 2)
    r = np.sqrt(dx * dx + dy * dy)
    phase = 0.5 + 0.5 * np.sin(2 * np.pi * (t / 7.0))
    grad = (20 + 60 * (1.0 - r) * phase).astype(np.float32)
    noise = np.random.normal(0, 3, (h, w)).astype(np.float32)
    img = np.clip(grad + noise, 0, 255).astype(np.uint8)
    return np.stack([img, img, img], axis=2)


def scene_from_img(path: Path, dur: float):
    clip = ImageClip(str(path)).resize(height=1920).set_duration(dur)
    w, h = clip.size
    return clip.fx(crop, x1=0, y1=0, x2=w, y2=h).resize(lambda t: 1.02 + 0.01 * t)


def scene_procedural(dur: float):
    return VideoClip(lambda t: procedural_frame(t), duration=dur).resize((1080, 1920))


# ==========================================
# === YOUTUBE UPLOAD ===
# ==========================================
def youtube_service():
    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
    token_uri = "https://oauth2.googleapis.com/token"
    creds = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri=token_uri,
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(path_mp4, title, desc, tags):
    youtube = youtube_service()
    body = {
        "snippet": {"title": title, "description": desc, "tags": tags, "categoryId": "24"},
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(path_mp4, chunksize=-1, resumable=True)
    return youtube.videos().insert(part=",".join(body.keys()), body=body, media_body=media).execute().get("id")


# ==========================================
# === ORCHESTRATOR ===
# ==========================================
def run_pipeline():
    uid = uuid.uuid4().hex[:6]
    script_text, scene_caps = make_script(total_sentences=random.randint(6, 9))
    raw_path = str(OUT / f"narr_{uid}.mp3")
    gTTS(text=script_text, lang="en").save(raw_path)
    voice_path = tts_and_design(script_text, raw_path)

    fetched = fetch_backgrounds(limit=8)
    bg_files = list(OVERLAYS.glob("*.jpg")) + list(OVERLAYS.glob("*.png"))
    scene_count = random.randint(SCENES_MIN, SCENES_MAX)
    per_scene = VIDEO_DURATION / scene_count
    clips = []
    for i in range(scene_count):
        base = scene_from_img(random.choice(bg_files), per_scene) if bg_files else scene_procedural(per_scene)
        cap = scene_caps[i % len(scene_caps)]
        cap_img = caption_png(cap, width=1000)
        cap_clip = ImageClip(cap_img).set_duration(per_scene).set_position(("center", "center"))
        clips.append(CompositeVideoClip([base, cap_clip]))

    final = clips[0]
    for c in clips[1:]:
        final = CompositeVideoClip([final.set_end(final.duration), c.set_start(final.duration)])
    final = final.set_duration(VIDEO_DURATION)
    aclip = AudioFileClip(voice_path)
    final = final.set_audio(aclip)

    mp4 = str(OUT / f"signal_auto_{uid}.mp4")
    final.write_videofile(mp4, fps=30, codec="libx264", audio_codec="aac")

    evolve_and_store(scene_caps)
    n = datetime.utcnow().strftime("%Y-%m-%d")
    title = f"SIGNAL ARCHIVE — {n} — {random.choice(['The Pulse Beneath','Whisper Archive','Eclipsera'])}"
    desc = (
        "From the T-3012 archives. God is not gone. He's buried.\n\n"
        "Automated longform transmission."
    )
    tags = ["godfinders", "ai", "cosmic horror", "existential", "shorts"]
    vid = upload_to_youtube(mp4, title, desc, tags)
    print("Uploaded video ID:", vid)


if __name__ == "__main__":
    run_pipeline()
