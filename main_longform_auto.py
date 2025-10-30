# main_longform_auto.py
import os, random, uuid, textwrap, re
from pathlib import Path
from datetime import datetime

import numpy as np
import requests
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip, VideoClip,
    concatenate_videoclips
)
from moviepy.video.fx.all import crop, fadein  # crossfadein yok, MoviePy 1.0.3 uyumu

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
SCENE_ASSETS = DATA / "scene_assets"
AUTOSEEDS = DATA / "autoseeds.txt"
OUT = ROOT / "out"
for p in (DATA, OVERLAYS, SCENE_ASSETS, OUT):
    p.mkdir(exist_ok=True)

LANG = os.getenv("LANG", "en")
TARGET_W, TARGET_H = 720, 1280
VIDEO_DURATION_CAP = int(os.getenv("VIDEO_DURATION", "120"))
SCENES_MIN = int(os.getenv("SCENES_MIN", "7"))
SCENES_MAX = int(os.getenv("SCENES_MAX", "10"))

# =========================
# SEEDS / SCRIPT
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
    "God": "the divine", "light": "signal", "darkness": "static",
    "soil": "veil", "bones": "wires", "blood": "current",
    "song": "frequency", "prayer": "protocol",
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
    return list(dict.fromkeys(lines))

def mutate_line(s: str) -> str:
    words = s.split()
    if not words: return s
    for i, w in enumerate(words):
        key = w.strip(".,;:!?\"'")
        if key in REPLACE_MAP and random.random() < 0.25:
            words[i] = w.replace(key, REPLACE_MAP[key])
    if random.random() < 0.15 and len(words) > 4:
        j = random.randrange(1, len(words)-1)
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
    """Hook → Discovery(2) → Escalation(2) → Revelation(2) → Tag"""
    bank = load_seeds()
    pick = lambda n: random.sample(bank, min(n, len(bank)))
    hook = random.choice(HOOKS)
    discovery = pick(2); escalation = pick(2); revelation = pick(2)
    tag = random.choice([
        "God is not gone. He’s buried.",
        "The Pulse counts backwards.",
        "No maps survived the light.",
        "Silence replied in numbers.",
    ])
    pieces = [hook] + discovery + escalation + revelation + [tag]
    full = " ".join(" ".join(pieces).split())
    if len(full) > 900: full = full[:897] + "…"
    caps = [hook] + [s if len(s) <= 90 else s[:87]+"…" for s in (discovery+escalation+revelation)] + [tag]
    return full, caps

# =========================
# SIMPLE KEYWORDING
# =========================
KW_MAP = {
    r"\b(pulse|signal|frequency|static)\b": ["oscilloscope", "astronomy radio", "deep space", "waveform"],
    r"\b(god|divine|prayer|cathedral|ritual)\b": ["cathedral ruin", "ancient temple", "monastery night"],
    r"\b(soil|desert|sand|veil)\b": ["desert night", "dune", "barren landscape"],
    r"\b(light|eclipse|star|sun)\b": ["eclipse", "stars", "galaxy", "nebula"],
    r"\b(bones|wires|machine|protocol)\b": ["factory interior", "industrial ruin", "circuit board macro"],
    r"\b(whisper|silence|numbers)\b": ["library dark", "crypt", "void"],
    r"\b(map|ruin|archive)\b": ["ruins", "old manuscript", "ancient map"],
}

def extract_keywords(line: str):
    qs = []
    for pat, choices in KW_MAP.items():
        if re.search(pat, line, flags=re.I):
            qs += choices
    if not qs:
        qs = ["eclipse", "alien landscape", "ruins night", "galaxy", "nebula"]
    return list(dict.fromkeys(qs))[:3]

# =========================
# AUDIO DESIGN (PER-CHUNK)
# =========================
def _pitch_down(seg, semitones=-2.5):
    new_fr = int(seg.frame_rate * (2.0 ** (semitones / 12.0)))
    return seg._spawn(seg.raw_data, overrides={"frame_rate": new_fr}).set_frame_rate(seg.frame_rate)

def _echo(seg, delay_ms=240, decay_db=-10):
    echo_part = seg - 6
    echo_part = AudioSegment.silent(duration=delay_ms) + (echo_part + decay_db)
    return seg.overlay(echo_part)

def _drone(duration_ms, freq=62.0, vol_db=-24):
    sr = 44100
    t = np.linspace(0, duration_ms/1000.0, int(sr*duration_ms/1000.0), endpoint=False)
    wave = (np.sin(2*np.pi*freq*t)*32767).astype(np.int16).tobytes()
    return AudioSegment(data=wave, sample_width=2, frame_rate=sr, channels=1) + vol_db

def tts_chunks_and_concatenate(lines):
    """
    Her sahne için ayrı TTS üret → gerçek süreleri al → concat et.
    Döner: combined_path, durations_sec(list), per_chunk_paths(list)
    """
    chunk_paths, durations = [], []
    for i, txt in enumerate(lines):
        path = OUT / f"tts_{i:02d}_{uuid.uuid4().hex[:4]}.mp3"
        gTTS(text=txt, lang="en").save(str(path))
        seg = AudioSegment.from_file(path).set_channels(1).set_frame_rate(44100)
        seg = _pitch_down(seg, -2.5)
        seg = _echo(seg, 200, -9)
        if len(seg) < 3000:
            seg = seg + AudioSegment.silent(duration=3000 - len(seg))
        seg.export(path, format="mp3")
        chunk_paths.append(str(path))
        durations.append(len(seg) / 1000.0)

    combined = AudioSegment.silent(duration=0)
    for p in chunk_paths:
        combined += AudioSegment.from_file(p)
    drone = _drone(len(combined), freq=random.choice([49.0, 62.0, 73.0]), vol_db=-26)
    mix = _echo(drone.overlay(combined), 410, -14)

    out_path = str(OUT / f"voice_{uuid.uuid4().hex[:6]}.mp3")
    mix.export(out_path, format="mp3")
    return out_path, durations, chunk_paths

# =========================
# CAPTION PNG
# =========================
def caption_png(text: str, width=900, padding=26):
    wrapped = textwrap.fill(text, 28)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 54)
    except Exception:
        font = ImageFont.load_default()
    dummy = Image.new("RGBA", (width, 10), (0,0,0,0))
    d = ImageDraw.Draw(dummy)
    bbox = d.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=6)
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    img = Image.new("RGBA", (width+2*padding, h+2*padding), (0,0,0,0))
    # gölge
    sd = ImageDraw.Draw(img)
    for ox, oy in ((1,1),(2,2),(-1,1),(1,-1),(-2,-2)):
        sd.multiline_text((img.width//2+ox, padding+oy), wrapped, font=font, fill=(0,0,0,160),
                          align="center", anchor="ma", spacing=6)
    # beyaz
    d2 = ImageDraw.Draw(img)
    d2.multiline_text((img.width//2, padding), wrapped, font=font, fill=(255,255,255,255),
                      align="center", anchor="ma", spacing=6)
    out = str(OUT / f"cap_{uuid.uuid4().hex[:6]}.png")
    img.save(out)
    return out

# =========================
# WIKIMEDIA FETCH PER SCENE
# =========================
COMMONS_ENDPOINT = "https://commons.wikimedia.org/w/api.php"

def fetch_one_image(query: str, w=TARGET_W, timeout=10):
    params = {
        "action":"query","format":"json","generator":"search",
        "gsrsearch":query,"gsrlimit":"1","gsrnamespace":"6",
        "prop":"imageinfo","iiprop":"url","iiurlwidth":str(w)
    }
    try:
        r = requests.get(COMMONS_ENDPOINT, params=params, timeout=timeout); r.raise_for_status()
        data = r.json(); pages = data.get("query", {}).get("pages", {})
        for _, p in pages.items():
            info = p.get("imageinfo", [{}])[0]
            url = info.get("thumburl") or info.get("url")
            if url and url.lower().endswith((".jpg",".jpeg",".png")):
                raw = requests.get(url, timeout=timeout); raw.raise_for_status()
                name = f"{uuid.uuid4().hex[:8]}_{query.replace(' ','_')}.jpg"
                path = SCENE_ASSETS / name
                path.write_bytes(raw.content)
                return str(path)
    except Exception:
        return None
    return None

def gen_procedural_still(path: Path, w=TARGET_W, h=TARGET_H):
    y, x = np.ogrid[:h, :w]
    cy, cx = h/2, w/2
    r = np.sqrt(((x-cx)/(w/2))**2 + ((y-cy)/(h/2))**2)
    band = (np.sin(r*10)*0.5+0.5)*180 + 30
    noise = np.random.normal(0, 6, (h, w))
    img = np.clip(band+noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(img).convert("RGB").filter(ImageFilter.GaussianBlur(1.2))
    vign = Image.new("L",(w,h),0); vd = ImageDraw.Draw(vign)
    vd.ellipse((-int(w*0.2),-int(h*0.2),int(w*1.2),int(h*1.2)), fill=255)
    vign = vign.filter(ImageFilter.GaussianBlur(80))
    im.putalpha(vign)
    bg = Image.new("RGB",(w,h),(8,8,10)); bg.paste(im, mask=im.split()[-1])
    bg.save(path)

def ensure_scene_images_for_line(line: str, need=2):
    queries = extract_keywords(line)
    paths = []
    for q in queries:
        got = fetch_one_image(q)
        if got: paths.append(got)
        if len(paths) >= need: break
    while len(paths) < need:
        tmp = SCENE_ASSETS / f"proc_{uuid.uuid4().hex[:6]}.jpg"
        gen_procedural_still(tmp)
        paths.append(str(tmp))
    return paths[:need]

# =========================
# SCENE BUILDERS
# =========================
def kenburns_clip(img_path: str, dur: float):
    clip = ImageClip(img_path).resize(height=TARGET_H).set_duration(dur)
    w, h = clip.size
    z = clip.fx(crop, x1=0, y1=0, x2=w, y2=h).resize(lambda t: 1.03 + 0.01 * t)
    return z

def scene_from_images(img_paths, dur: float):
    """
    2 görsel → her biri yarım süre.
    İkincisi ~0.6 sn 'fade-in' alır ve bir öncekinin son 0.6 sn'si üzerine bindirilir
    → MoviePy 1.0.3 ile yumuşak crossfade etkisi.
    """
    if len(img_paths) == 1:
        return kenburns_clip(img_paths[0], dur)

    half = max(0.8, (dur / 2.0))
    xfade = min(0.6, half * 0.4)

    a = kenburns_clip(img_paths[0], half).set_start(0)
    b = kenburns_clip(img_paths[1], half)
    b = fadein(b, xfade).set_start(half - xfade)

    comp = CompositeVideoClip([a, b], size=(TARGET_W, TARGET_H))
    comp = comp.set_duration(half + half)
    return comp

# =========================
# YOUTUBE
# =========================
def youtube_service():
    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
    token_uri = "https://oauth2.googleapis.com/token"
    creds = Credentials(
        None, refresh_token=refresh_token, token_uri=token_uri,
        client_id=client_id, client_secret=client_secret,
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

    # 1) Metin + sahne başlıkları
    full_text, caps_all = make_script_longform()
    target_scene_count = max(5, min(len(caps_all), random.randint(SCENES_MIN, SCENES_MAX)))
    caps = caps_all[:target_scene_count]

    # 2) TTS her sahne için ayrı → concat → süre listesi
    voice_path, durations, per_chunk = tts_chunks_and_concatenate(caps)
    total_voice_sec = sum(durations)
    if total_voice_sec > VIDEO_DURATION_CAP:
        overflow = total_voice_sec - VIDEO_DURATION_CAP
        if durations[-1] > overflow + 1.0:
            durations[-1] -= overflow
            total_voice_sec = VIDEO_DURATION_CAP
        else:
            durations.pop(); caps.pop()
            total_voice_sec = sum(durations)

    # 3) Her sahneye uygun görselleri getir
    all_scene_clips = []
    t_cursor = 0.0
    for i, (cap, dur) in enumerate(zip(caps, durations)):
        imgs = ensure_scene_images_for_line(cap, need=2)
        base = scene_from_images(imgs, dur).set_start(t_cursor)

        cap_img = caption_png(cap, width=900)
        cap_clip = ImageClip(cap_img).set_duration(dur).set_start(t_cursor).set_position(("center","center"))

        all_scene_clips += [base, cap_clip]
        t_cursor += dur

    # 4) Composite + Audio
    timeline = CompositeVideoClip(all_scene_clips, size=(TARGET_W, TARGET_H))
    aclip = AudioFileClip(voice_path)
    timeline = timeline.set_audio(aclip).set_duration(total_voice_sec)

    # 5) Render
    mp4 = str(OUT / f"signal_auto_{uid}.mp4")
    timeline.write_videofile(
        mp4, fps=24, codec="libx264", audio_codec="aac",
        bitrate="2500k", threads=2,
        ffmpeg_params=["-preset", "veryfast", "-tune", "stillimage"]
    )

    # 6) Seed evrimi
    evolve_and_store(caps)

    # 7) YouTube upload — TARİH YOK, sadece başlık
    title = f"SIGNAL ARCHIVE — {random.choice(['The Pulse Beneath','Whisper Archive','Eclipsera'])}"
    desc = (
        "From the T-3012 archives. God is not gone. He's buried.\n\n"
        "Automated longform transmission."
    )
    tags = ["godfinders","ai","cosmic horror","existential","longform"]

    vid = upload_to_youtube(mp4, title, desc, tags)
    print("Uploaded video ID:", vid)


if __name__ == "__main__":
    run_pipeline()
