"""
AUTOYUSON — Factual Shorts (History / Space / Culture) for Yusonvasya
- Wikipedia'dan doğrulanabilir özet + kaynak URL
- gTTS ile seslendirme
- PIL ile metin (başlık + altyazı) PNG oluşturma (ImageMagick YOK)
- MoviePy ile 1080x1920 video
- YouTube'a yükleme (Education=27), başlıkta tarih YOK
- Konu tekrarını data/seen_topics.json ile azaltır

ENV (GitHub Secrets):
  YT_CLIENT_ID
  YT_CLIENT_SECRET
  YT_REFRESH_TOKEN

Opsiyonel ENV:
  LANGUAGE=en
  VIDEO_DURATION=55
  TITLE_PREFIX=
  CONTENT_THEME=FACTUAL_SHORTS
  TOPICS_FILE=topics.txt
"""

import os, io, re, json, random, textwrap, pathlib, sys, tempfile
from typing import List, Tuple, Optional
from dataclasses import dataclass

import requests
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip, afx
from moviepy.video.fx.all import resize as mp_resize
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import wikipedia

# ---------- Config ----------
LANG = os.getenv("LANGUAGE", "en").strip() or "en"
DURATION_TARGET = int(os.getenv("VIDEO_DURATION", "55"))
TITLE_PREFIX = os.getenv("TITLE_PREFIX", "").strip()
YOUTUBE_CATEGORY_ID = "27"
DEFAULT_TAGS = ["history", "space", "astronomy", "archaeology", "culture", "documentary", "facts", "shorts"]
APPEND_DATE_TO_TITLE = False
TOPICS_FILE = os.getenv("TOPICS_FILE", "topics.txt")

wikipedia.set_lang("en")

CURATED_TOPICS = [
    "Rosetta Stone","Dead Sea Scrolls","Hagia Sophia","Voyager Golden Record","Terracotta Army","Göbekli Tepe",
    "Library of Alexandria","Code of Hammurabi","Magna Carta","Edict of Milan","Battle of Thermopylae",
    "Byzantine Iconoclasm","Hubble's law","Pioneer 10","James Webb Space Telescope","Apollo 8 Earthrise",
    "Arecibo message","Cassini–Huygens","Venera 13","Voyager 1","Chandrasekhar limit","Nazca Lines",
    "Antikythera mechanism","I Ching","Shahnameh","Silk Road","Machu Picchu","Moai","Çatalhöyük"
]

DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
SEEN_PATH = DATA_DIR / "seen_topics.json"
FONTS_DIR = pathlib.Path("fonts"); FONTS_DIR.mkdir(exist_ok=True)

W, H = 1080, 1920

# ---------- Seen topics ----------
def load_seen() -> List[str]:
    if SEEN_PATH.exists():
        try: return json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        except Exception: return []
    return []

def save_seen(seen: List[str]) -> None:
    SEEN_PATH.write_text(json.dumps(seen[-1000:], ensure_ascii=False, indent=2), encoding="utf-8")

def load_extra_topics() -> List[str]:
    p = pathlib.Path(TOPICS_FILE)
    if p.exists():
        return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    return []

def pick_topic(seen: List[str]) -> str:
    pool = load_extra_topics() + CURATED_TOPICS
    random.shuffle(pool)
    recent = set(seen[-300:])
    for t in pool:
        if t not in recent:
            return t
    return random.choice(pool)

# ---------- Wikipedia ----------
def wiki_fetch(topic: str) -> Tuple[str, str, str, Optional[str]]:
    try:
        page = wikipedia.page(topic, auto_suggest=False, preload=True)
    except Exception:
        hits = wikipedia.search(topic)
        if not hits: raise RuntimeError("No wiki results")
        page = wikipedia.page(hits[0], auto_suggest=True, preload=True)
    title = page.title
    summary = wikipedia.summary(title, sentences=4)
    url = page.url
    image_url = None
    for img in page.images:
        low = img.lower()
        if any(b in low for b in ["logo","icon","flag","seal","map","diagram","coat_of_arms"]): continue
        if low.endswith((".jpg",".jpeg",".png")):
            image_url = img; break
    return title, summary, url, image_url

def download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=30); r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

# ---------- Script ----------
def craft_script(title: str, summary: str, sources: List[str]) -> Tuple[str, str]:
    txt = re.sub(r"\[\d+\]", "", summary).strip()
    txt = re.sub(r"\s+", " ", txt)
    words = txt.split()
    if len(words) > 120: body = " ".join(words[:120])
    elif len(words) < 80: body = txt + " " + " ".join((words+words)[:80])
    else: body = txt
    hook = f"{title}: a verified chapter of our shared past."
    takeaway = "What endures is evidence — and what it reveals."
    narration = f"{hook}\n{body}\n{takeaway}\n\nSources:\n" + "\n".join(f"- {s}" for s in sources[:4])
    return narration, title

def is_factual_enough(text: str) -> bool:
    if "sources:" not in text.lower(): return False
    if "http" not in text.lower(): return False
    bad = ["reportedly","allegedly","some say","legend has it","it is said","rumor"]
    if any(w in text.lower() for w in bad): return False
    n = len(text.split()); return 80 <= n <= 200

# ---------- Fonts (no ImageMagick) ----------
N_REG = FONTS_DIR / "NotoSans-Regular.ttf"
N_BOLD = FONTS_DIR / "NotoSans-Bold.ttf"

def ensure_fonts():
    # Google Fonts kaynakları
    srcs = {
        N_REG: "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
        N_BOLD:"https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf"
    }
    for path, url in srcs.items():
        if not path.exists():
            try:
                r = requests.get(url, timeout=30); r.raise_for_status()
                path.write_bytes(r.content)
            except Exception:
                pass  # PIL default fonta düşer

def get_font(size=48, bold=False):
    try:
        return ImageFont.truetype(str(N_BOLD if bold else N_REG), size=size)
    except Exception:
        return ImageFont.load_default()

# ---------- Text to PNG ----------
def pil_text_image(text: str, width: int, fontsize: int, bold: bool=False, fill=(255,255,255), padding=8) -> Image.Image:
    font = get_font(size=fontsize, bold=bold)
    # kelime sarmalama
    lines = []
    cur = ""
    draw = ImageDraw.Draw(Image.new("RGB",(10,10)))
    for word in text.split():
        test = (cur + " " + word).strip()
        w, _ = draw.textsize(test, font=font)
        if w > width - 2*padding:
            if cur: lines.append(cur)
            cur = word
        else:
            cur = test
    if cur: lines.append(cur)
    # boyut
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + 6
    img_h = padding*2 + line_h*len(lines)
    img = Image.new("RGBA", (width, img_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    y = padding
    for ln in lines:
        w,_ = draw.textsize(ln, font=font)
        x = (width - w)//2
        draw.text((x, y), ln, font=font, fill=fill)
        y += line_h
    return img

def save_text_png(text: str, out_path: str, width: int, fontsize: int, bold=False, fill=(255,255,255)):
    img = pil_text_image(text, width=width, fontsize=fontsize, bold=bold, fill=fill)
    img.save(out_path, "PNG")

def wrap_lines_for_subs(text: str, max_chars=52) -> List[str]:
    lines = []
    for para in text.splitlines():
        p = para.strip()
        if not p or p.lower().startswith("sources:"): continue
        chunk = textwrap.fill(p, width=max_chars)
        lines.extend([l for l in chunk.split("\n") if l.strip()])
    return lines

# ---------- Render ----------
@dataclass
class RenderResult:
    video_path: str
    title: str
    description: str
    tags: List[str]

def fit_center_crop(img: Image.Image, w=W, h=H) -> Image.Image:
    ratio_bg = img.width / img.height
    ratio_frame = w / h
    if ratio_bg > ratio_frame:
        new_h = h; new_w = int(ratio_bg * new_h)
    else:
        new_w = w; new_h = int(new_w / ratio_bg)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (img.width - w)//2; top = (img.height - h)//2
    return img.crop((left, top, left+w, top+h))

def render_video(title: str, narration: str, bg_img: Image.Image, audio_path: str, duration_hint: int) -> RenderResult:
    ensure_fonts()
    bg = fit_center_crop(bg_img)
    bg_path = "bg_autoyuson.jpg"; bg.save(bg_path, quality=92)

    narration_audio = AudioFileClip(audio_path)
    duration = max(min(max(duration_hint, 40), 75), narration_audio.duration + 1.0)

    bg_clip = ImageClip(bg_path).set_duration(duration).fx(mp_resize, 1.06)

    # Title (PNG)
    title_png = "title.png"
    save_text_png(title[:64], title_png, width=W-140, fontsize=72, bold=True)
    title_clip = ImageClip(title_png).set_duration(4.0).set_position(("center", 80)).fadein(0.3).fadeout(0.3)

    # Subtitles (PNGs)
    subs = []
    sub_lines = wrap_lines_for_subs(narration, max_chars=46)
    per_line = max(2.0, min(4.0, duration / max(8, len(sub_lines) or 8)))
    t = 1.0
    for line in sub_lines:
        fn = f"sub_{len(subs):03d}.png"
        save_text_png(line, fn, width=W-160, fontsize=50, bold=False)
        clip = ImageClip(fn).set_start(t).set_duration(per_line).set_position(("center", H-420)).fadein(0.15).fadeout(0.15)
        subs.append(clip)
        t += per_line * 0.9

    narration_audio = narration_audio.fx(afx.audio_normalize)
    comp_audio = CompositeAudioClip([narration_audio.set_start(0)])
    comp = CompositeVideoClip([bg_clip, title_clip, *subs]).set_audio(comp_audio).set_duration(duration)

    out_name = re.sub(r"[^-\w\s.,()]+","",title).strip().replace(" ","_")[:80] or "autoyuson_video"
    out_path = f"{out_name}.mp4"
    comp.write_videofile(out_path, fps=30, codec="libx264", audio_codec="aac",
                         temp_audiofile="temp-audio.m4a", remove_temp=True, threads=4, verbose=False, logger=None)

    # Description (sources)
    sources_block = []
    capture = False
    for line in narration.splitlines():
        if line.strip().lower().startswith("sources:"):
            capture = True; continue
        if capture and line.strip().startswith("-"):
            sources_block.append(line.strip()[2:])
    description = (f"{title}\n\nShort factual video. No speculation, no fiction.\n\n"
                   f"Sources:\n" + "\n".join(f"- {s}" for s in sources_block[:6]) +
                   "\n\n#history #space #culture #facts #shorts")

    return RenderResult(video_path=out_path, title=title, description=description, tags=DEFAULT_TAGS)

# ---------- YouTube ----------
def build_youtube_service():
    client_id = os.environ["YT_CLIENT_ID"]
    client_secret = os.environ["YT_CLIENT_SECRET"]
    refresh_token = os.environ["YT_REFRESH_TOKEN"]
    creds = Credentials(token=None, refresh_token=refresh_token,
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=client_id, client_secret=client_secret,
                        scopes=["https://www.googleapis.com/auth/youtube.upload","https://www.googleapis.com/auth/youtube"])
    creds.refresh(Request())
    return build("youtube","v3",credentials=creds)

def youtube_upload(video_path: str, title: str, description: str, tags: List[str],
                   category_id: str = YOUTUBE_CATEGORY_ID, privacy_status: str = "public") -> str:
    from googleapiclient.http import MediaFileUpload
    youtube = build_youtube_service()
    body = {"snippet":{"title": title if not TITLE_PREFIX else f"{TITLE_PREFIX}{title}",
                       "description": description, "categoryId": category_id, "tags": tags},
            "status":{"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False}}
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/*")
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
    return resp.get("id","")

# ---------- Orchestrator ----------
def tts_to_file(text: str, outpath: str, lang: str = "en"):
    gTTS(text=text, lang=lang, tld="com").save(outpath)

def main() -> int:
    seen = load_seen()
    topic = pick_topic(seen)
    print(f"[AUTOYUSON] Topic: {topic}")

    title, summary, url, img_url = wiki_fetch(topic)
    sources = [url]
    narration, short_title = craft_script(title, summary, sources)
    if not is_factual_enough(narration):
        summary2 = wikipedia.summary(title, sentences=5)
        narration, short_title = craft_script(title, summary2, sources)

    # Background image
    if img_url:
        try: img = download_image(img_url)
        except Exception: img = Image.new("RGB",(W,H),(20,20,24))
    else:
        img = Image.new("RGB",(W,H),(20,20,24))

    audio_path = "narration_autoyuson.mp3"
    tts_to_file(narration, audio_path, LANG)

    result = render_video(short_title, narration, img, audio_path, DURATION_TARGET)

    print("[AUTOYUSON] Uploading...")
    vid = youtube_upload(result.video_path, result.title, result.description, result.tags)
    print(f"[AUTOYUSON] Uploaded: https://youtube.com/watch?v={vid}")

    seen.append(title); save_seen(seen)
    print("[AUTOYUSON] Seen topics updated.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
