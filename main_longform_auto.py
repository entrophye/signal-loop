# main_longform_auto.py
import os, random, uuid, textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import requests
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    VideoClip,
)
from moviepy.video.fx.all import crop

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request


# =========================
# ENV & PATHS
# =========================
load_dotenv()
ROOT = Path(__file__).parent
DATA = ROOT / "data"
OVERLAYS = DATA / "overlays"
AUTOSEEDS = DATA / "autoseeds.txt"
OUT = ROOT / "out"
for p in (DATA, OVERLAYS, OUT):
    p.mkdir(exist_ok=True)

LANG = os.getenv("LANG", "en")

# Render hedefi (dikey)
TARGET_W, TARGET_H = 720, 1280

# Süre ve sahne konfigleri
VIDEO_DURATION = int(os.getenv("VIDEO_DURATION", "120"))  # üst sınır/niyet
SCENES_MIN = int(os.getenv("SCENES_MIN", "7"))
SCENES_MAX = int(os.getenv("SCENES_MAX", "10"))

# =========================
# SEED / METİN
# =========================
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

HOOKS = [
    "Transmission begins.",
    "Archive fragment recovered.",
    "Field log: Eclipsera — Sector Hell’s Eve.",
    "After the collapse, we listened.",
]

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
    # unique (sıra korunur)
    return list(dict.fromkeys(lines))


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


def make_script_longform():
    """
    Mini 3-akt yapı:
    Hook → Discovery (2) → Escalation (2) → Revelation (2) → Tag (imza)
    TTS için tek parça 'full' metin, sahne başlıkları için kısa 'scene_caps'.
    """
    bank = load_seeds()
    pick = lambda n: random.sample(bank, min(n, len(bank)))

    hook = random.choice(HOOKS)
    discovery = pick(2)
    escalation = pick(2)
    revelation = pick(2)
    tag = random.choice([
        "God is not gone. He’s buried.",
        "The Pulse counts backwards.",
        "No maps survived the light.",
        "Silence replied in numbers.",
    ])

    paragraphs = [hook] + discovery + escalation + revelation + [tag]
    full = " ".join(paragraphs)
    full = " ".join(full.split())
    if len(full) > 900:
        full = full[:897] + "…"

    caps = []
    caps.append(hook)
    for s in (discovery + escalation + revelation):
        caps.append(s if len(s) <= 90 else (s[:87] + "…"))
    caps.append(tag)
    return full, caps


# =========================
# AUDIO TASARIM
# =========================
def _pitch_down(seg, semitones=-2.5):
    new_fr = int(seg.frame_rate * (2.0 ** (semitones / 12.0)))
    return seg._spawn(seg.raw_data, overrides={"frame_rate": new_fr}).set_frame_rate(seg.frame_rate)


def _echo(seg, delay_ms=240, decay_db=-10):
    echo_part = seg - 6
    echo_part = AudioSegment.silent(duration=delay_ms) + (echo_part + decay_db)
    return seg.overlay(echo_part)


def _drone(duration_ms, freq=62.0, vol_db=-24):
    # Basit sinüs drone'u (mono)
    sr = 44100
    t = np.linspace(0, duration_ms / 1000.0, int(sr * duration_ms / 1000.0), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16).tobytes()
    return AudioSegment(data=wave, sample_width=2, frame_rate=sr, channels=1) + vol_db


def tts_and_design(text, raw_mp3_path):
    """
    raw_mp3_path: gTTS çıktısı (ham)
    Dönen: tasarlanmış ses dosyası (drone + pitch + echo) yolu
    """
    v = AudioSegment.from_file(raw_mp3_path).set_channels(1).set_frame_rate(44100)
    v = _pitch_down(v, -2.5)
    v = _echo(v, 240, -10)

    # Hedef süre: ses, VIDEO_DURATION'ı aşabilir; sorun değil.
    d = _drone(len(v), freq=random.choice([49.0, 62.0, 73.0]), vol_db=-25)
    mix = _echo(d.overlay(v), 410, -14)

    out2 = str(OUT / f"voice_{uuid.uuid4().hex[:6]}.mp3")
    mix.export(out2, format="mp3")
    return out2


# =========================
# CAPTION PNG
# =========================
def caption_png(text: str, width=900, padding=26):
    wrapped = textwrap.fill(text, 28)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 54)
    except Exception:
        font = ImageFont.load_default()

    dummy = Image.new("RGBA", (width, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    bbox = d.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=6)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    img = Image.new("RGBA", (width + 2 * padding, h + 2 * padding), (0, 0, 0, 0))
    # Shadow stroke
    sd = ImageDraw.Draw(img)
    for ox, oy in ((1, 1), (2, 2), (-1, 1), (1, -1), (-2, -2)):
        sd.multiline_text(
            (img.width // 2 + ox, padding + oy),
            wrapped,
            font=font,
            fill=(0, 0, 0, 160),
            align="center",
            anchor="ma",
            spacing=6,
        )
    # White text
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


# =========================
# BACKGROUND FETCH/PROCEDURAL
# =========================
COMMONS_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
KEYWORDS = [
    "nebula", "galaxy", "desert night", "cathedral ruin",
    "alien landscape", "stars", "eclipse", "ruins night", "cave dark",
]


def fetch_backgrounds(limit=10, timeout=12):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": "|".join(KEYWORDS),
        "gsrlimit": str(limit),
        "gsrnamespace": "6",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": str(TARGET_W),
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

        # eski overlayleri temizle
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


def gen_procedural_still(path: Path, w=TARGET_W, h=TARGET_H):
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    r = np.sqrt(((x - cx) / (w / 2)) ** 2 + ((y - cy) / (h / 2)) ** 2)
    band = (np.sin(r * 10) * 0.5 + 0.5) * 180 + 30
    noise = np.random.normal(0, 6, (h, w))
    img = np.clip(band + noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(img).convert("RGB").filter(ImageFilter.GaussianBlur(1.2))
    # hafif vignette
    vign = Image.new("L", (w, h), 0)
    vd = ImageDraw.Draw(vign)
    vd.ellipse((-int(w*0.2), -int(h*0.2), int(w*1.2), int(h*1.2)), fill=255)
    vign = vign.filter(ImageFilter.GaussianBlur(80))
    im.putalpha(vign)
    # arkaplan siyah ile birleştir
    bg = Image.new("RGB", (w, h), (8, 8, 10))
    bg.paste(im, mask=im.split()[-1])
    bg.save(path)


def ensure_backgrounds(min_count=8):
    count = fetch_backgrounds(limit=max(min_count, 10))
    files = list(OVERLAYS.glob("*.jpg")) + list(OVERLAYS.glob("*.png"))
    if len(files) >= min_count:
        return
    need = min_count - len(files)
    for i in range(need):
        gen_procedural_still(OVERLAYS / f"proc_{i}.jpg")


# =========================
# VIDEO SCENES
# =========================
def scene_from_img(path: Path, dur: float):
    clip = ImageClip(str(path)).resize(height=TARGET_H).set_duration(dur)
    w, h = clip.size
    # yumuşak zoom-in
    return clip.fx(crop, x1=0, y1=0, x2=w, y2=h).resize(lambda t: 1.02 + 0.01 * t)


def scene_procedural(dur: float):
    # Kare başı numpy üretimi pahalı; burada sabit procedural still + yavaş zoom:
    tmp = OVERLAYS / f"proc_static_{uuid.uuid4().hex[:6]}.jpg"
    gen_procedural_still(tmp)
    return scene_from_img(tmp, dur)


# =========================
# YOUTUBE UPLOAD
# =========================
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


# =========================
# ORCHESTRATOR
# =========================
def run_pipeline():
    uid = uuid.uuid4().hex[:6]

    # 1) Metin (daha dramatik uzun form)
    script_text, scene_caps = make_script_longform()

    # 2) TTS (ham) + ses tasarım
    raw_path = str(OUT / f"narr_{uid}.mp3")
    gTTS(text=script_text, lang="en").save(raw_path)
    voice_path = tts_and_design(script_text, raw_path)

    # Gerçek ses süresi (sn) → sahneleri buna göre böleceğiz
    voice_ms = len(AudioSegment.from_file(voice_path))
    voice_sec = max(10, voice_ms // 1000)  # alt sınır 10 sn
    scene_count = max(5, min(len(scene_caps), random.randint(SCENES_MIN, SCENES_MAX)))
    per_scene = voice_sec / scene_count

    # 3) Arkaplanları garanti et
    ensure_backgrounds(min_count=8)
    bg_files = list(OVERLAYS.glob("*.jpg")) + list(OVERLAYS.glob("*.png"))

    # 4) Zaman çizelgesi — her sahneyi START/END ile yerleştir (senkron altyazı)
    timeline_elements = []
    for i in range(scene_count):
        start_t = i * per_scene
        dur = per_scene

        base = (scene_from_img(random.choice(bg_files), dur) if bg_files else scene_procedural(dur)).set_start(start_t)

        cap = scene_caps[i % len(scene_caps)]
        cap_img = caption_png(cap, width=900)
        cap_clip = (
            ImageClip(cap_img)
            .set_duration(dur)
            .set_start(start_t)
            .set_position(("center", "center"))
        )

        timeline_elements.append(base)
        timeline_elements.append(cap_clip)

    # 5) Composite + Ses
    timeline = CompositeVideoClip(timeline_elements, size=(TARGET_W, TARGET_H))
    aclip = AudioFileClip(voice_path)
    # Sesin süresini baz al; video süresi maksimum VIDEO_DURATION ile sınırla (gerekirse kısalt)
    total_dur = min(voice_sec, VIDEO_DURATION)
    timeline = timeline.set_audio(aclip).set_duration(total_dur)

    # 6) Render (hız/kalite dengesi)
    mp4 = str(OUT / f"signal_auto_{uid}.mp4")
    timeline.write_videofile(
        mp4,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        bitrate="2500k",
        threads=2,
        ffmpeg_params=["-preset", "veryfast", "-tune", "stillimage"],
    )

    # 7) Seed evrimi
    evolve_and_store(scene_caps)

    # 8) YouTube upload
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
