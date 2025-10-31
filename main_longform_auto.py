# main_longform_auto.py
import os, random, uuid, textwrap, re, hashlib
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip, VideoClip
)
from moviepy.video.fx.all import crop, fadein  # MoviePy 1.0.3 uyumlu

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request


# ============== ENV & PATHS ==============
load_dotenv()
ROOT = Path(__file__).parent
DATA = ROOT / "data"
SCENE_ASSETS = DATA / "scene_assets"
AUTOSEEDS = DATA / "autoseeds.txt"
HISTORY = DATA / "history.log"
OUT = ROOT / "out"
for p in (DATA, SCENE_ASSETS, OUT):
    p.mkdir(exist_ok=True)

LANG = os.getenv("LANG", "en")
TARGET_W, TARGET_H = 720, 1280
ASPECT = TARGET_W / TARGET_H
VIDEO_DURATION_CAP = int(os.getenv("VIDEO_DURATION", "120"))
SCENES_MIN = int(os.getenv("SCENES_MIN", "7"))
SCENES_MAX = int(os.getenv("SCENES_MAX", "10"))

# ============== SEEDS / SCRIPT ==============
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
    "The archive listens back.",
    "Numbers bled into shapes; shapes learned to pray.",
    "We drew a circle around the last signal and stepped inside."
]
HOOKS = [
    "Transmission begins.",
    "Archive fragment recovered.",
    "Field log: Eclipsera — Sector Hell's Eve.",
    "After the collapse, we listened."
]
REPLACE_MAP = {
    "God": "the divine", "light": "signal", "darkness": "static",
    "soil": "veil", "bones": "wires", "blood": "current",
    "song": "frequency", "prayer": "protocol",
}

TITLE_ADJ = [
    "Whisper", "Pulse", "Static", "Axiom", "Veil", "Cathedral",
    "Depth", "Relay", "Ashen", "Hollow", "Archive", "Orbit"
]
TITLE_CORE = [
    "Archive", "Transmission", "Relay", "Protocol", "Litany",
    "Signal", "Hypostasis", "Strain", "Breath", "Canticle"
]

def load_seeds():
    lines = []
    if AUTOSEEDS.exists():
        for s in AUTOSEEDS.read_text(encoding="utf-8").splitlines():
            s = s.strip()
            if s:
                lines.append(s)
    if not lines:
        lines = BASE_SEEDS[:]
    # benzersiz sırayı koru
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

def _hash_signature(lines):
    sig = "||".join(lines).encode("utf-8", errors="ignore")
    return hashlib.sha256(sig).hexdigest()

def _history_load():
    if not HISTORY.exists():
        return []
    return [ln.strip() for ln in HISTORY.read_text(encoding="utf-8").splitlines() if ln.strip()]

def _history_save(sig):
    hist = _history_load()
    hist.append(sig)
    HISTORY.write_text("\n".join(hist[-200:]), encoding="utf-8")

def make_script_longform():
    """
    Dinamik şablonlar + mutasyon + sıra karması.
    Çıkan satırların hash'i history ile karşılaştırılır; çakışırsa yeniden üretilir.
    Dönen:
      - full (kısa kullanım için)
      - caps (sahne başlıkları dizisi)
    """
    bank = load_seeds()
    rng = random.Random(uuid.uuid4().int)  # her koşuda farklı

    TEMPLATES = [
        "{hook} {a}. {b}. {c}. {tag}",
        "{hook} {a}. {b}. {c}. {d}. {tag}",
        "{hook} {a}. Then {b}. {c}. {tag}",
        "{hook} {a}. {b}. We listened. {c}. {tag}",
    ]

    # İçerik havuzu—mutasyon + karmayla
    base = rng.sample(bank, min(len(bank), 12))
    pool = [mutate_line(s) for s in base]
    rng.shuffle(pool)

    def build_once():
        hook = rng.choice(HOOKS)
        picks = pool[: rng.randint(6, 9)]
        rng.shuffle(picks)
        # kısa sahne başlıkları (caps)
        caps = [hook] + [p[:90] + ("…" if len(p) > 90 else "") for p in picks[:6]] + [rng.choice([
            "God is not gone. He's buried.",
            "The Pulse counts backwards.",
            "Silence replied in numbers.",
            "The archive listens back."
        ])]
        # full metin
        tpl = rng.choice(TEMPLATES)
        def take_or_blank(i): return picks[i] if i < len(picks) else ""
        full = tpl.format(
            hook=hook, a=take_or_blank(0), b=take_or_blank(1),
            c=take_or_blank(2), d=take_or_blank(3),
            tag=caps[-1]
        )
        full = re.sub(r"\s+", " ", full).strip()
        sig = _hash_signature(caps)
        return full, caps, sig

    history = set(_history_load())
    for _ in range(6):
        full, caps, sig = build_once()
        if sig not in history:
            _history_save(sig)
            return full, caps
    _history_save(sig)
    return full, caps

# ============== KEYWORDING ==============
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

# ============== AUDIO (PER-CHUNK) ==============
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

# ============== CAPTION PNG (auto-fit) ==============
def caption_png(text: str, screen_w=TARGET_W, screen_h=TARGET_H, max_w_ratio=0.86, max_h_ratio=0.55):
    padding = 28
    max_w = int(screen_w * max_w_ratio)
    max_h = int(screen_h * max_h_ratio)
    try_order = [64, 60, 56, 52, 48, 44, 40, 36, 32]

    base_font = "DejaVuSans.ttf"

    def render_try(font_px):
        try:
            font = ImageFont.truetype(base_font, font_px)
        except Exception:
            font = ImageFont.load_default()

        # Dinamik satır kırma
        words = text.split()
        lines, current = [], []
        dummy = Image.new("RGBA", (max_w, 10), (0, 0, 0, 0))
        d = ImageDraw.Draw(dummy)

        for w in words:
            test = (" ".join(current + [w])).strip()
            bbox = d.multiline_textbbox((0, 0), test, font=font)
            if (bbox[2] - bbox[0]) <= max_w:
                current.append(w)
            else:
                if current:
                    lines.append(" ".join(current))
                    current = [w]
                else:
                    lines.append(w)
                    current = []
        if current:
            lines.append(" ".join(current))

        text_h = 0
        line_metrics = []
        for ln in lines:
            bb = d.multiline_textbbox((0, 0), ln, font=font)
            lw, lh = bb[2] - bb[0], bb[3] - bb[1]
            line_metrics.append((ln, lw, lh))
            text_h += lh

        img_w = min(max_w, max((m[1] for m in line_metrics), default=0)) + 2 * padding
        img_h = text_h + 2 * padding + (6 * max(0, len(lines) - 1))

        fits = (img_w <= max_w + 2 * padding) and (img_h <= max_h + 2 * padding)
        return fits, font, line_metrics, img_w, img_h

    chosen = None
    for px in try_order:
        fits, font, lines, img_w, img_h = render_try(px)
        if fits:
            chosen = (px, font, lines, img_w, img_h)
            break
    if not chosen:
        px, font, lines, img_w, img_h = render_try(try_order[-1])
    else:
        px, font, lines, img_w, img_h = chosen

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    total_text_h = sum(lh for _, _, lh in lines) + (6 * max(0, len(lines) - 1))
    y = (img_h - total_text_h) // 2

    for ln, lw, lh in lines:
        x = (img_w - lw) // 2
        for ox, oy in ((1, 1), (2, 2), (-1, 1), (1, -1), (-2, -2)):
            d.text((x + ox, y + oy), ln, font=font, fill=(0, 0, 0, 160))
        d.text((x, y), ln, font=font, fill=(255, 255, 255, 255))
        y += lh + 6

    out = str(OUT / f"cap_{uuid.uuid4().hex[:6]}.png")
    img.save(out)
    return out

# ============== IMAGE FETCH & PROCEDURAL ==============
COMMONS_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
HTTP_HEADERS = {
    "User-Agent": "SignalLoopBot/1.0 (contact: example@example.com)",
    "Accept": "application/json",
    "Referer": "https://commons.wikimedia.org/",
}

def fetch_one_image(query: str, w=TARGET_W, timeout=10):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": "1",
        "gsrnamespace": "6",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": str(w),
        "origin": "*",
    }
    try:
        r = requests.get(COMMONS_ENDPOINT, params=params, headers=HTTP_HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for _, p in pages.items():
            info = p.get("imageinfo", [{}])[0]
            url = info.get("thumburl") or info.get("url")
            if url and url.lower().endswith((".jpg", ".jpeg", ".png")):
                raw = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
                raw.raise_for_status()
                safe = re.sub(r"[^a-zA-Z0-9_]+", "_", query)
                name = f"{uuid.uuid4().hex[:8]}_{safe}.jpg"
                path = SCENE_ASSETS / name
                path.write_bytes(raw.content)
                print(f"[image] fetched: {query} -> {name}")
                return str(path)
    except Exception as e:
        print(f"[image] fetch fail: {query} ({e})")
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
    # vignette
    vign = Image.new("L",(w,h),0); vd = ImageDraw.Draw(vign)
    vd.ellipse((-int(w*0.2),-int(h*0.2),int(w*1.2),int(h*1.2)), fill=255)
    vign = vign.filter(ImageFilter.GaussianBlur(80))
    im.putalpha(vign)
    bg = Image.new("RGB",(w,h),(8,8,10)); bg.paste(im, mask=im.split()[-1])
    bg.save(path)

def ensure_scene_images_for_line(line: str, need=2):
    queries = extract_keywords(line)
    print(f"[scene] keywords for '{line[:40]}...': {queries}")
    paths = []
    for q in queries:
        got = fetch_one_image(q)
        if got: paths.append(got)
        if len(paths) >= need: break
    while len(paths) < need:
        tmp = SCENE_ASSETS / f"proc_{uuid.uuid4().hex[:6]}.jpg"
        gen_procedural_still(tmp)
        paths.append(str(tmp))
        print(f"[image] procedural: {tmp.name}")
    return paths[:need]

# ============== SCENE BUILDERS ==============
def fit_and_fill(clip: ImageClip):
    w, h = clip.size
    in_aspect = w / h
    if in_aspect > ASPECT:
        clip = clip.resize(height=TARGET_H)
        w2, h2 = clip.size
        new_w = int(TARGET_H * ASPECT)
        x1 = (w2 - new_w) // 2
        clip = crop(clip, x1=x1, y1=0, x2=x1+new_w, y2=TARGET_H)
    else:
        clip = clip.resize(width=TARGET_W)
        w2, h2 = clip.size
        new_h = int(TARGET_W / ASPECT)
        y1 = (h2 - new_h) // 2
        clip = crop(clip, x1=0, y1=y1, x2=TARGET_W, y2=y1+new_h)
    return clip.set_duration(clip.duration)

def kenburns_clip(img_path: str, dur: float):
    clip = ImageClip(img_path).set_duration(dur)
    clip = fit_and_fill(clip)
    return clip.resize(lambda t: 1.03 + 0.01 * t)

def animated_fallback_bg(duration: float):
    def make_frame(t):
        y = np.linspace(0, 1, TARGET_H).reshape(-1, 1)
        x = np.linspace(0, 1, TARGET_W).reshape(1, -1)
        band = (np.sin(2*np.pi*(y*1.2 + 0.08*t)) * 0.5 + 0.5)
        band2 = (np.cos(2*np.pi*(x*1.0 - 0.06*t)) * 0.5 + 0.5)
        base = (band*0.6 + band2*0.4)
        noise = np.random.normal(0, 0.02, (TARGET_H, TARGET_W))
        img = np.clip((base + noise) * 30 + 8, 0, 255).astype(np.uint8)
        frame = np.stack([img, img, img], axis=2)
        return frame
    return VideoClip(make_frame, duration=duration)

def scene_from_images(img_paths, dur: float):
    xfade = 0.6
    bg = animated_fallback_bg(dur).set_start(0)

    if not img_paths:
        return bg

    if len(img_paths) == 1:
        a = kenburns_clip(img_paths[0], dur).set_start(0)
        comp = CompositeVideoClip([bg, a], size=(TARGET_W, TARGET_H)).set_duration(dur)
        return comp

    half = max(0.8, (dur / 2.0))
    a = kenburns_clip(img_paths[0], half).set_start(0)
    b = kenburns_clip(img_paths[1], half)
    b = fadein(b, xfade).set_start(half - xfade)

    comp = CompositeVideoClip([bg, a, b], size=(TARGET_W, TARGET_H))
    comp = comp.set_duration(half + half)
    return comp

# ============== YOUTUBE ==============
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

# ============== ORCHESTRATOR ==============
def run_pipeline():
    uid = uuid.uuid4().hex[:6]

    # 1) Script
    _, caps_all = make_script_longform()
    target_scene_count = max(5, min(len(caps_all), random.randint(SCENES_MIN, SCENES_MAX)))
    caps = caps_all[:target_scene_count]

    # 2) TTS → süre listesi
    voice_path, durations, _ = tts_chunks_and_concatenate(caps)
    total_voice_sec = sum(durations)
    if total_voice_sec > VIDEO_DURATION_CAP:
        overflow = total_voice_sec - VIDEO_DURATION_CAP
        if durations[-1] > overflow + 1.0:
            durations[-1] -= overflow
            total_voice_sec = VIDEO_DURATION_CAP
        else:
            durations.pop(); caps.pop()
            total_voice_sec = sum(durations)

    # 3) Görseller + captions
    all_scene_clips = []
    t_cursor = 0.0
    for i, (cap, dur) in enumerate(zip(caps, durations)):
        imgs = ensure_scene_images_for_line(cap, need=2)
        base = scene_from_images(imgs, dur).set_start(t_cursor)

        cap_img = caption_png(cap, screen_w=TARGET_W, screen_h=TARGET_H, max_w_ratio=0.86, max_h_ratio=0.55)
        cap_clip = ImageClip(cap_img).set_duration(dur).set_start(t_cursor).set_position(("center", "center"))

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

    # 7) YouTube (tarih yok; her koşuda değişken başlık)
    title = "SIGNAL ARCHIVE — " + random.choice(TITLE_ADJ) + " " + random.choice(TITLE_CORE)

    lead_options = [
        "Recovered long-range transmission.",
        "Field log stitched from T-3012 archives.",
        "A relay caught between static and prayer."
    ]
    lead = random.choice(lead_options)

    desc = (
        lead + "\n\n"
        "God is not gone. He's buried.\n"
        "Automated longform transmission."
    )

    tags = ["godfinders", "ai", "cosmic horror", "existential", "longform", "eclipsera", "signal"]

    vid = upload_to_youtube(mp4, title, desc, tags)
    print("Uploaded video ID:", vid)


if __name__ == "__main__":
    run_pipeline()
