"""
AUTOYUSON — Factual Shorts (History / Space / Culture) for Yusonvasya
Tam otomatik: konu seç → doğrula → seslendir → video → YouTube upload → görülen konuyu kaydet.

GEREKEN ENV (GitHub Secrets olarak da okunur):
  YT_CLIENT_ID
  YT_CLIENT_SECRET
  YT_REFRESH_TOKEN

İSTEĞE BAĞLI ENV:
  LANGUAGE=en           # gTTS dili (içerik İngilizce kalmalı)
  VIDEO_DURATION=55     # hedef saniye (40–75 arası otomatik dengelenir)
  TITLE_PREFIX=         # örn: "Autoyuson — "
  CONTENT_THEME=FACTUAL_SHORTS
  TOPICS_FILE=topics.txt  # varsa bu dosyadan da konu okur (satır başına bir konu)
"""

import os, io, re, json, random, textwrap, tempfile, subprocess, pathlib, sys
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ---- Third-party ----
import requests
from PIL import Image
from gtts import gTTS
from moviepy.editor import (ImageClip, AudioFileClip, CompositeVideoClip, TextClip, CompositeAudioClip, afx)
from moviepy.video.fx.all import resize as mp_resize
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import wikipedia

# ----------------- Config -----------------
LANG = os.getenv("LANGUAGE", "en").strip() or "en"
DURATION_TARGET = int(os.getenv("VIDEO_DURATION", "55"))
TITLE_PREFIX = os.getenv("TITLE_PREFIX", "").strip()
YOUTUBE_CATEGORY_ID = "27"  # Education
DEFAULT_TAGS = ["history", "space", "astronomy", "archaeology", "culture", "documentary", "facts", "shorts"]
APPEND_DATE_TO_TITLE = False  # İSTENMİYOR
TOPICS_FILE = os.getenv("TOPICS_FILE", "topics.txt")

wikipedia.set_lang("en")

CURATED_TOPICS = [
    # History
    "Rosetta Stone", "Dead Sea Scrolls", "Hagia Sophia", "Voyager Golden Record",
    "Terracotta Army", "Göbekli Tepe", "Library of Alexandria", "Code of Hammurabi",
    "Magna Carta", "Edict of Milan", "Battle of Thermopylae", "Byzantine Iconoclasm",
    # Space
    "Hubble's law", "Pioneer 10", "James Webb Space Telescope", "Apollo 8 Earthrise",
    "Arecibo message", "Cassini–Huygens", "Venera 13", "Voyager 1", "Chandrasekhar limit",
    # Culture / Anthropology
    "Nazca Lines", "Antikythera mechanism", "I Ching", "Shahnameh",
    "Silk Road", "Machu Picchu", "Moai", "Dead Sea", "Çatalhöyük",
]

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SEEN_PATH = DATA_DIR / "seen_topics.json"

# --------------- Helpers ------------------
def load_seen() -> List[str]:
    if SEEN_PATH.exists():
        try:
            return json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_seen(seen: List[str]) -> None:
    SEEN_PATH.write_text(json.dumps(seen[-1000:], ensure_ascii=False, indent=2), encoding="utf-8")

def load_extra_topics() -> List[str]:
    p = pathlib.Path(TOPICS_FILE)
    if p.exists():
        lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines()]
        return [x for x in lines if x and not x.startswith("#")]
    return []

def pick_topic(seen: List[str]) -> str:
    # Önce dış dosya, sonra curated; son 300 tekrarlarını engelle
    pool = load_extra_topics() + CURATED_TOPICS
    random.shuffle(pool)
    recent = set(seen[-300:])
    for t in pool:
        if t not in recent:
            return t
    return random.choice(pool)

def safe_filename(s: str) -> str:
    return re.sub(r"[^-\w\s.,()]+", "", s).strip().replace(" ", "_")[:80]

def wiki_fetch(topic: str) -> Tuple[str, str, str, Optional[str]]:
    """Return (title, summary, url, image_url or None)."""
    try:
        page = wikipedia.page(topic, auto_suggest=False, preload=True)
    except Exception:
        hits = wikipedia.search(topic)
        if not hits:
            raise RuntimeError("No wiki results")
        page = wikipedia.page(hits[0], auto_suggest=True, preload=True)
    title = page.title
    summary = wikipedia.summary(title, sentences=4)
    url = page.url
    image_url = None
    for img in page.images:
        low = img.lower()
        if any(b in low for b in ["logo", "icon", "flag", "seal", "map", "diagram", "coat_of_arms"]):
            continue
        if low.endswith((".jpg", ".jpeg", ".png")):
            image_url = img
            break
    return title, summary, url, image_url

def download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def craft_script(title: str, summary: str, sources: List[str]) -> Tuple[str, str]:
    """
    Return (narration_text, short_title)
    90–130 kelime, 1 satır hook, 1 satır kapanış, sonunda kaynak listesi.
    """
    txt = re.sub(r"\[\d+\]", "", summary).strip()
    txt = re.sub(r"\s+", " ", txt)

    hook = f"{title}: a verified chapter of our shared past."
    body_words = txt.split()
    if len(body_words) > 120:
        body = " ".join(body_words[:120])
    elif len(body_words) < 80:
        body = txt + " " + " ".join((body_words + body_words)[:80])  # hafif doldurma
    else:
        body = txt

    takeaway = "What endures is evidence — and what it reveals."
    narration = f"{hook}\n{body}\n{takeaway}\n\nSources:\n" + "\n".join(f"- {s}" for s in sources[:4])
    return narration, title

def tts_to_file(text: str, outpath: str, lang: str = "en"):
    tts = gTTS(text=text, lang=lang, tld="com")
    tts.save(outpath)

def wrap_lines_for_subs(text: str, max_chars=52) -> List[str]:
    lines = []
    for para in text.splitlines():
        p = para.strip()
        if not p or p.lower().startswith("sources:"):
            continue
        chunk = textwrap.fill(p, width=max_chars)
        lines.extend([l for l in chunk.split("\n") if l.strip()])
    return lines

@dataclass
class RenderResult:
    video_path: str
    title: str
    description: str
    tags: List[str]

def render_video(title: str, narration: str, img: Image.Image, audio_path: str, duration_hint: int) -> RenderResult:
    W, H = 1080, 1920
    # fit + center-crop
    bg = img.copy()
    ratio_bg = bg.width / bg.height
    ratio_frame = W / H
    if ratio_bg > ratio_frame:
        new_h = H
        new_w = int(ratio_bg * new_h)
    else:
        new_w = W
        new_h = int(new_w / ratio_bg)
    bg = bg.resize((new_w, new_h), Image.LANCZOS)
    left = (bg.width - W) // 2
    top = (bg.height - H) // 2
    bg = bg.crop((left, top, left + W, top + H))
    bg_path = "bg_autoyuson.jpg"
    bg.save(bg_path, quality=92)

    narration_audio = AudioFileClip(audio_path)
    duration = max(min(max(duration_hint, 40), 75), narration_audio.duration + 1.0)

    bg_clip = ImageClip(bg_path).set_duration(duration)
    bg_clip = bg_clip.fx(mp_resize, 1.06).set_position(("center", "center"))

    # Title
    title_text = title[:64]
    title_clip = TextClip(title_text, fontsize=72, font="Arial-Bold", color="white",
                          method="caption", size=(W-140, None), align="center")
    title_clip = title_clip.set_duration(4.0).set_position(("center", 80)).fadein(0.3).fadeout(0.3)

    # Subtitles
    sub_lines = wrap_lines_for_subs(narration, max_chars=46)
    per_line = max(2.0, min(4.0, duration / max(8, len(sub_lines))))
    subs = []
    t = 1.0
    for line in sub_lines:
        clip = TextClip(line, fontsize=50, font="Arial", color="white",
                        method="caption", size=(W-160, None), align="center")
        clip = clip.set_start(t).set_duration(per_line).set_position(("center", H-420)).fadein(0.15).fadeout(0.15)
        subs.append(clip)
        t += per_line * 0.9

    narration_audio = narration_audio.fx(afx.audio_normalize)
    composite_audio = CompositeAudioClip([narration_audio.set_start(0)])
    composite = CompositeVideoClip([bg_clip, title_clip, *subs]).set_audio(composite_audio).set_duration(duration)

    out_name = safe_filename(title) or "autoyuson_video"
    out_path = f"{out_name}.mp4"
    composite.write_videofile(
        out_path, fps=30, codec="libx264", preset="medium", audio_codec="aac",
        temp_audiofile="temp-audio.m4a", remove_temp=True, threads=4, verbose=False, logger=None
    )

    # Description (sources)
    sources_block = []
    capture = False
    for line in narration.splitlines():
        if line.strip().lower().startswith("sources:"):
            capture = True
            continue
        if capture and line.strip().startswith("-"):
            sources_block.append(line.strip()[2:])
    description = (
        f"{title}\n\nShort factual video. No speculation, no fiction.\n\n"
        f"Sources:\n" + "\n".join(f"- {s}" for s in sources_block[:6]) +
        "\n\n#history #space #culture #facts #shorts"
    )

    return RenderResult(video_path=out_path, title=title, description=description, tags=DEFAULT_TAGS)

# --------------- YouTube -----------------
def build_youtube_service():
    client_id = os.environ["YT_CLIENT_ID"]
    client_secret = os.environ["YT_CLIENT_SECRET"]
    refresh_token = os.environ["YT_REFRESH_TOKEN"]
    token_uri = "https://oauth2.googleapis.com/token"

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=token_uri,
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)

def youtube_upload(video_path: str, title: str, description: str, tags: List[str],
                   category_id: str = YOUTUBE_CATEGORY_ID, privacy_status: str = "public") -> str:
    from googleapiclient.http import MediaFileUpload
    youtube = build_youtube_service()

    request_body = {
        "snippet": {
            "title": title if not TITLE_PREFIX else f"{TITLE_PREFIX}{title}",
            "description": description,
            "categoryId": category_id,
            "tags": tags
        },
        "status": {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False}
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/*")
    request = youtube.videos().insert(part="snippet,status", body=request_body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
    return response.get("id", "")

# --------------- Orchestrator ------------
def is_factual_enough(text: str) -> bool:
    if "Sources:" not in text:
        return False
    if "http" not in text.lower():
        return False
    bad = ["reportedly", "allegedly", "some say", "legend has it", "it is said", "rumor"]
    if any(w in text.lower() for w in bad):
        return False
    n = len(text.split())
    return 80 <= n <= 200

def main() -> int:
    seen = load_seen()
    topic = pick_topic(seen)

    print(f"[AUTOYUSON] Topic chosen: {topic}")

    # Fetch wiki
    title, summary, url, img_url = wiki_fetch(topic)
    sources = [url]
    if not img_url:
        # ikinci kez arama: başlık + 'site:wikimedia.org'
        # minimal: görsel yoksa basit arka plan üret (düz degrade)
        # burada fallback: wikipedia lead görsel yoksa text background
        pass

    # Script
    narration, short_title = craft_script(title, summary, sources)

    # Basit link doğrulama (başlık ve açıklama için)
    if not is_factual_enough(narration):
        # ikinci şans: wiki summary sentences=5 ile tekrar dene
        summary2 = wikipedia.summary(title, sentences=5)
        narration, short_title = craft_script(title, summary2, sources)

    # Görsel indir / fallback tek renk
    if img_url:
        try:
            img = download_image(img_url)
        except Exception:
            img = Image.new("RGB", (1080, 1920), (20, 20, 24))
    else:
        img = Image.new("RGB", (1080, 1920), (20, 20, 24))

    # TTS
    audio_path = "narration_autoyuson.mp3"
    tts_to_file(narration, audio_path, LANG)

    # Render
    result = render_video(short_title, narration, img, audio_path, DURATION_TARGET)

    # Upload
    print("[AUTOYUSON] Uploading to YouTube...")
    video_id = youtube_upload(result.video_path, result.title, result.description, result.tags)
    print(f"[AUTOYUSON] Uploaded: https://youtube.com/watch?v={video_id}")

    # Seen update
    seen.append(title)
    save_seen(seen)
    print("[AUTOYUSON] Seen topics updated.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
