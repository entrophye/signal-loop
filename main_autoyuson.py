"""
AUTOYUSON — Factual Shorts (History / Space / Culture) for Yusonvasya

Güncellemeler:
- python-wikipedia kaldırıldı. Doğrudan Wikipedia REST + Media API kullanımı:
  • /api/rest_v1/page/summary/{title}
  • /api/rest_v1/page/media/{title}
  • /w/api.php?action=opensearch ... (fallback arama)
- Konu adı bozuk gelse bile OpenSearch ile en iyi eşleşme bulunur.
- Edge-TTS (Neural) ses, gTTS fallback.
- Cümle bazlı altyazı + süre orantılama.
- Çoklu görsel (Ken Burns + crossfade).
- ImageMagick yok; tüm metinler PIL ile PNG.

ENV (GitHub Secrets veya local env):
  # Birincil yol:
  YT_CLIENT_ID
  YT_CLIENT_SECRET
  YT_REFRESH_TOKEN

  # Yedek yol (bunlar boşsa çalışır):
  TOKEN_JSON_BASE64   # token.json içeriğinin base64 hâli (refresh_token + client_id + client_secret)

Opsiyonel:
  LANGUAGE=en
  VIDEO_DURATION=55
  TITLE_PREFIX=
  CONTENT_THEME=FACTUAL_SHORTS
  TOPICS_FILE=topics.txt
  EDGE_TTS_VOICE=en-US-AriaNeural
"""

import os, io, re, json, base64, random, textwrap, pathlib, sys, asyncio, urllib.parse
from typing import List, Tuple, Optional
from dataclasses import dataclass

import requests
from PIL import Image, ImageDraw, ImageFont

# Pillow 10+ uyumluluk şimi
try:
    _ = Image.ANTIALIAS  # type: ignore[attr-defined]
except AttributeError:
    try:
        Image.ANTIALIAS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except Exception:
        pass

from gtts import gTTS
from moviepy.editor import (ImageClip, AudioFileClip, CompositeVideoClip,
                            CompositeAudioClip, afx)
from moviepy.video.fx.all import resize as mp_resize
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# ---------- Config ----------
LANG = os.getenv("LANGUAGE", "en").strip() or "en"
DURATION_TARGET = int(os.getenv("VIDEO_DURATION", "55"))
TITLE_PREFIX = os.getenv("TITLE_PREFIX", "").strip()
YOUTUBE_CATEGORY_ID = "27"
DEFAULT_TAGS = ["history", "space", "astronomy", "archaeology", "culture", "documentary", "facts", "shorts"]
APPEND_DATE_TO_TITLE = False
TOPICS_FILE = os.getenv("TOPICS_FILE", "topics.txt")
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural")

DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
SEEN_PATH = DATA_DIR / "seen_topics.json"
FONTS_DIR = pathlib.Path("fonts"); FONTS_DIR.mkdir(exist_ok=True)

W, H = 1080, 1920
WIKI = "https://en.wikipedia.org"
REST = f"{WIKI}/api/rest_v1"
API  = f"{WIKI}/w/api.php"
UA = {"User-Agent": "autoyuson-bot/1.0 (https://github.com/)"}  # nazik olalım

CURATED_TOPICS = [
    "Rosetta Stone","Dead Sea Scrolls","Hagia Sophia","Voyager Golden Record","Terracotta Army","Göbekli Tepe",
    "Library of Alexandria","Code of Hammurabi","Magna Carta","Edict of Milan","Battle of Thermopylae",
    "Byzantine Iconoclasm","Hubble's law","Pioneer 10","James Webb Space Telescope","Apollo 8 Earthrise",
    "Arecibo message","Cassini–Huygens","Venera 13","Voyager 1","Chandrasekhar limit","Nazca Lines",
    "Antikythera mechanism","I Ching","Shahnameh","Silk Road","Machu Picchu","Moai","Çatalhöyük"
]

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

# ---------- Wikipedia (REST) ----------
def _rest_summary(title: str):
    url = f"{REST}/page/summary/{urllib.parse.quote(title)}"
    r = requests.get(url, headers=UA, timeout=20)
    if r.status_code != 200:
        return None
    return r.json()

def _rest_media(title: str):
    url = f"{REST}/page/media/{urllib.parse.quote(title)}"
    r = requests.get(url, headers=UA, timeout=20)
    if r.status_code != 200:
        return None
    return r.json()

def _opensearch(q: str) -> List[str]:
    r = requests.get(API, headers=UA, timeout=20, params={
        "action": "opensearch",
        "search": q,
        "limit": 5,
        "namespace": 0,
        "format": "json"
    })
    if r.status_code != 200:
        return []
    data = r.json()
    return data[1] if isinstance(data, list) and len(data) >= 2 else []

def wiki_fetch(topic: str):
    """
    Robust: önce summary dener, 404 olursa opensearch ile en iyi başlığı bulur,
    tekrar summary + media çağırır.

    Return: (title, summary, page_url, image_urls:list)
    """
    cand_titles = [topic] + _opensearch(topic)
    for cand in cand_titles:
        s = _rest_summary(cand)
        if not s: continue
        # summary & url
        title = s.get("title") or cand
        extract = s.get("extract") or ""
        page_url = s.get("content_urls", {}).get("desktop", {}).get("page") or f"{WIKI}/wiki/{urllib.parse.quote(title.replace(' ','_'))}"
        # media
        imgs: List[str] = []
        m = _rest_media(title)
        if m and "items" in m:
            for it in m["items"]:
                # Type "image" olanlar
                if it.get("type") != "image":
                    continue
                # source / original
                src = None
                if "srcset" in it and it["srcset"]:
                    # en büyük boyu al
                    src = sorted(it["srcset"], key=lambda x: x.get("scale", 1.0))[-1].get("src")
                if not src:
                    src = it.get("src")
                if not src:
                    continue
                low = str(src).lower()
                if any(b in low for b in ["logo","icon","flag","seal","map","diagram","coat_of_arms","favicon","sprite"]):
                    continue
                if not low.endswith((".jpg",".jpeg",".png",".webp")):
                    continue
                imgs.append(src)
        # fallback: summary thumb
        thumb = s.get("thumbnail", {}).get("source")
        if thumb:
            imgs = [thumb] + imgs
        if extract:
            # başarı
            return title, extract, page_url, imgs[:8]
    # hiçbir şey bulunamazsa hata
    raise RuntimeError(f"Wiki fetch failed for topic: {topic}")

def download_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, headers=UA, timeout=30); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return img
    except Exception:
        return None

# ---------- Script ----------
def craft_script(title: str, summary: str, page_url: str) -> Tuple[str, str, List[str]]:
    txt = re.sub(r"\s+", " ", summary).strip()
    words = txt.split()
    if len(words) > 120: body = " ".join(words[:120])
    elif len(words) < 80: body = txt + " " + " ".join((words+words)[:80])
    else: body = txt
    hook = f"{title}: a verified chapter of our shared past."
    takeaway = "What endures is evidence — and what it reveals."
    narration = f"{hook} {body} {takeaway}"
    # cümlelere böl
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', narration) if s.strip()]
    return narration, title, sentences

def is_factual_enough(text: str) -> bool:
    bad = ["reportedly","allegedly","some say","legend has it","it is said","rumor"]
    if any(w in text.lower() for w in bad): return False
    n = len(text.split()); return 80 <= n <= 240

# ---------- Fonts ----------
N_REG = FONTS_DIR / "NotoSans-Regular.ttf"
N_BOLD = FONTS_DIR / "NotoSans-Bold.ttf"

def ensure_fonts():
    srcs = {
        N_REG: "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
        N_BOLD:"https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf"
    }
    for path, url in srcs.items():
        if not path.exists():
            try:
                r = requests.get(url, headers=UA, timeout=30); r.raise_for_status()
                path.write_bytes(r.content)
            except Exception:
                pass  # default font

def get_font(size=48, bold=False):
    try:
        return ImageFont.truetype(str(N_BOLD if bold else N_REG), size=size)
    except Exception:
        return ImageFont.load_default()

# ---------- Text to PNG ----------
def pil_text_image(text: str, width: int, fontsize: int, bold: bool=False, fill=(255,255,255), padding=8) -> Image.Image:
    font = get_font(size=fontsize, bold=bold)
    lines, cur = [], ""
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
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + 6
    img_h = padding*2 + line_h*len(lines)
    img = Image.new("RGBA", (width, img_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    y = padding
    for ln in lines:
        w,_ = draw.textsize(ln, font=font)
        x = (width - w)//2
        draw.text((x+2, y+2), ln, font=font, fill=(0,0,0,160))
        draw.text((x, y), ln, font=font, fill=fill)
        y += line_h
    return img

def save_text_png(text: str, out_path: str, width: int, fontsize: int, bold=False, fill=(255,255,255)):
    img = pil_text_image(text, width=width, fontsize=fontsize, bold=bold, fill=fill)
    img.save(out_path, "PNG")

# ---------- TTS ----------
async def edge_tts_synthesize_async(text: str, outpath: str, voice: str):
    import edge_tts, aiofiles
    communicator = edge_tts.Communicate(text, voice=voice)
    async with aiofiles.open(outpath, "wb") as f:
        async for chunk in communicator.stream():
            if chunk["type"] == "audio":
                await f.write(chunk["data"])

def tts_to_file(text: str, outpath: str, lang: str = "en", voice: Optional[str] = None):
    try:
        if voice:
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            asyncio.get_event_loop().run_until_complete(edge_tts_synthesize_async(text, outpath, voice))
            return
    except Exception:
        pass
    # fallback
    gTTS(text=text, lang=lang, tld="com").save(outpath)

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

def build_broll_clips(bg_images: List[Image.Image], duration: float):
    if not bg_images:
        bg_images = [Image.new("RGB",(W,H),(20,20,24))]
    n = len(bg_images)
    per = max(3.5, min(8.0, duration / max(1, n)))
    clips = []
    for i, im in enumerate(bg_images):
        im_fit = fit_center_crop(im)
        path = f"bg_{i:02d}.jpg"
        im_fit.save(path, quality=92)
        clip = ImageClip(path).set_duration(per).fx(mp_resize, 1.06)
        clips.append(clip)
    # crossfade bindirmeli katman
    t = 0.0
    layers = []
    for i, c in enumerate(clips):
        d = c.duration
        fc = 0.5
        layers.append(c.set_start(t).crossfadein(fc).crossfadeout(fc))
        t += d - fc
    return CompositeVideoClip(layers).set_duration(duration)

def split_sentences_for_subs(sentences: List[str], audio_duration: float) -> List[Tuple[str, float]]:
    toks = [len(re.findall(r"\w+", s)) or 1 for s in sentences]
    total = sum(toks) or 1
    base = max(0.8, audio_duration * 0.92)
    return [(sentences[i], max(1.6, base * (toks[i] / total))) for i in range(len(sentences))]

def render_video(title: str, narration: str, sentences: List[str],
                 bg_images: List[Image.Image], audio_path: str, duration_hint: int) -> RenderResult:
    ensure_fonts()

    narration_audio = AudioFileClip(audio_path)
    duration = max(min(max(duration_hint, 40), 75), narration_audio.duration + 0.8)

    broll = build_broll_clips(bg_images, duration)

    # Title
    title_png = "title.png"
    save_text_png(title[:64], title_png, width=W-140, fontsize=72, bold=True)
    title_clip = ImageClip(title_png).set_duration(4.0).set_position(("center", 80)).fadein(0.3).fadeout(0.3)

    # Subtitles (cümle bazlı)
    per_lines = split_sentences_for_subs(sentences, narration_audio.duration)
    subs = []
    cur_t = 0.8
    for i, (line, dur) in enumerate(per_lines):
        fn = f"sub_{i:03d}.png"
        save_text_png(line, fn, width=W-160, fontsize=50, bold=False)
        clip = ImageClip(fn).set_start(cur_t).set_duration(dur).set_position(("center", H-420)).fadein(0.15).fadeout(0.15)
        subs.append(clip)
        cur_t += dur * 0.98

    narration_audio = narration_audio.fx(afx.audio_normalize)
    comp_audio = CompositeAudioClip([narration_audio.set_start(0)])
    comp = CompositeVideoClip([broll, title_clip, *subs]).set_audio(comp_audio).set_duration(duration)

    out_name = re.sub(r"[^-\w\s.,()]+","",title).strip().replace(" ","_")[:80] or "autoyuson_video"
    out_path = f"{out_name}.mp4"
    comp.write_videofile(out_path, fps=30, codec="libx264", audio_codec="aac",
                         temp_audiofile="temp-audio.m4a", remove_temp=True, threads=4, verbose=False, logger=None)

    # Description
    wiki_url = f"{WIKI}/wiki/{urllib.parse.quote(title.replace(' ','_'))}"
    description = (f"{title}\n\nShort factual video. No speculation, no fiction.\n\n"
                   f"Source:\n- Wikipedia: {wiki_url}"
                   "\n\n#history #space #culture #facts #shorts")
    return RenderResult(video_path=out_path, title=title, description=description, tags=DEFAULT_TAGS)

# ---------- OAuth helpers ----------
def _read_oauth_from_env_or_b64():
    cid = os.getenv("YT_CLIENT_ID", "").strip()
    csecret = os.getenv("YT_CLIENT_SECRET", "").strip()
    rtoken = os.getenv("YT_REFRESH_TOKEN", "").strip()
    if cid and csecret and rtoken:
        return cid, csecret, rtoken
    b64 = os.getenv("TOKEN_JSON_BASE64", "").strip()
    if not b64:
        return cid, csecret, rtoken
    try:
        raw = base64.b64decode(b64).decode("utf-8")
        data = json.loads(raw)
        rtoken2 = (data.get("refresh_token") or "").strip()
        cid2 = (data.get("client_id") or cid).strip()
        csecret2 = (data.get("client_secret") or csecret).strip()
        return cid2 or cid, csecret2 or csecret, rtoken2 or rtoken
    except Exception:
        return cid, csecret, rtoken

def build_youtube_service():
    client_id, client_secret, refresh_token = _read_oauth_from_env_or_b64()
    if not refresh_token:
        raise RuntimeError("Missing refresh_token. Set YT_REFRESH_TOKEN or TOKEN_JSON_BASE64.")
    if not client_id or not client_secret:
        cid_env = os.getenv("YT_CLIENT_ID", "").strip()
        cs_env  = os.getenv("YT_CLIENT_SECRET", "").strip()
        client_id = client_id or cid_env
        client_secret = client_secret or cs_env
        if not client_id or not client_secret:
            raise RuntimeError("Missing client_id/client_secret.")
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
def main() -> int:
    seen = load_seen()
    topic = pick_topic(seen)
    print(f"[AUTOYUSON] Topic: {topic}")

    title, summary, page_url, image_urls = wiki_fetch(topic)
    narration, short_title, sentences = craft_script(title, summary, page_url)
    if not is_factual_enough(narration):
        # summary zaten REST'ten; burada ek bir manevraya gerek yok
        pass

    # Görseller
    bg_images: List[Image.Image] = []
    for u in image_urls:
        im = download_image(u)
        if im: bg_images.append(im)
    if not bg_images:
        bg_images = [Image.new("RGB",(W,H),(20,20,24))]

    # TTS (Edge-TTS → gTTS fallback)
    audio_path = "narration_autoyuson.mp3"
    tts_to_file(narration, audio_path, LANG, voice=EDGE_TTS_VOICE)

    # Render
    result = render_video(short_title, narration, sentences, bg_images, audio_path, DURATION_TARGET)

    # Upload
    print("[AUTOYUSON] Uploading...")
    vid = youtube_upload(result.video_path, result.title, result.description, result.tags)
    print(f"[AUTOYUSON] Uploaded: https://youtube.com/watch?v={vid}")

    seen.append(title); save_seen(seen)
    print("[AUTOYUSON] Seen topics updated.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
