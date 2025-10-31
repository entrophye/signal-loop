"""
AUTOYUSON — Factual Shorts (History / Space / Culture) for Yusonvasya

Güncellemeler:
- Tek görselde bile tam süre Ken Burns (siyah ekran bitirildi)
- Çoklu görsel: crossfade + hafif zoom (belgesel hissi)
- Ambiyans müziği: procedural drone (veya assets/ambience.mp3 varsa onu kullanır)
- Kelime senkron altyazı: Edge-TTS word boundary varsa birebir; yoksa 2–3 kelimelik hızlı chunk’lar
- ImageMagick YOK; metin PNG’leri PIL ile

ENV (GitHub Secrets veya local):
  YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN
  # Yedek:
  TOKEN_JSON_BASE64   (token.json base64)

Opsiyonel:
  LANGUAGE=en
  VIDEO_DURATION=55
  TITLE_PREFIX=
  TOPICS_FILE=topics.txt
  EDGE_TTS_VOICE=en-US-AriaNeural
"""

import os, io, re, json, base64, random, textwrap, pathlib, sys, asyncio, urllib.parse, math
from typing import List, Tuple, Optional
from dataclasses import dataclass

import requests
from PIL import Image, ImageDraw, ImageFont

# Pillow 10+ uyumluluk
try:
    _ = Image.ANTIALIAS  # type: ignore[attr-defined]
except AttributeError:
    try:
        Image.ANTIALIAS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
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
TOPICS_FILE = os.getenv("TOPICS_FILE", "topics.txt")
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural")

DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
SEEN_PATH = DATA_DIR / "seen_topics.json"
FONTS_DIR = pathlib.Path("fonts"); FONTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR = pathlib.Path("assets"); ASSETS_DIR.mkdir(exist_ok=True)

W, H = 1080, 1920
WIKI = "https://en.wikipedia.org"
REST = f"{WIKI}/api/rest_v1"
API  = f"{WIKI}/w/api.php"
UA = {"User-Agent": "autoyuson-bot/1.1 (+github)"}

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
        "action": "opensearch", "search": q, "limit": 5, "namespace": 0, "format": "json"
    })
    if r.status_code != 200:
        return []
    data = r.json()
    return data[1] if isinstance(data, list) and len(data) >= 2 else []

def wiki_fetch(topic: str):
    cand_titles = [topic] + _opensearch(topic)
    for cand in cand_titles:
        s = _rest_summary(cand)
        if not s: continue
        title = s.get("title") or cand
        extract = s.get("extract") or ""
        page_url = s.get("content_urls", {}).get("desktop", {}).get("page") or f"{WIKI}/wiki/{urllib.parse.quote(title.replace(' ','_'))}"
        imgs: List[str] = []
        m = _rest_media(title)
        if m and "items" in m:
            for it in m["items"]:
                if it.get("type") != "image": continue
                src = None
                if it.get("srcset"):
                    src = sorted(it["srcset"], key=lambda x: x.get("scale",1.0))[-1].get("src")
                if not src: src = it.get("src")
                if not src: continue
                low = str(src).lower()
                if any(b in low for b in ["logo","icon","flag","seal","map","diagram","coat_of_arms","favicon","sprite"]): continue
                if not low.endswith((".jpg",".jpeg",".png",".webp")): continue
                imgs.append(src)
        thumb = s.get("thumbnail", {}).get("source")
        if thumb: imgs = [thumb] + imgs
        if extract:
            # en az 3 görsel hedefleyelim (yoksa tek görseli tekrar kullanırız)
            return title, extract, page_url, imgs[:8]
    raise RuntimeError(f"Wiki fetch failed for topic: {topic}")

def download_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, headers=UA, timeout=30); r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

# ---------- Script ----------
def craft_script(title: str, summary: str) -> Tuple[str, str, List[str]]:
    txt = re.sub(r"\s+", " ", summary).strip()
    words = txt.split()
    if len(words) > 120: body = " ".join(words[:120])
    elif len(words) < 80: body = txt + " " + " ".join((words+words)[:80])
    else: body = txt
    hook = f"{title}: a verified chapter of our shared past."
    takeaway = "What endures is evidence — and what it reveals."
    narration = f"{hook} {body} {takeaway}"
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', narration) if s.strip()]
    return narration, title, sentences

def is_factual_enough(text: str) -> bool:
    bad = ["reportedly","allegedly","some say","legend has it","it is said","rumor"]
    if any(w in text.lower() for w in bad): return False
    n = len(text.split()); return 80 <= n <= 240

# ---------- Fonts / Text PNG ----------
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
                pass

def get_font(size=48, bold=False):
    try:
        return ImageFont.truetype(str(N_BOLD if bold else N_REG), size=size)
    except Exception:
        return ImageFont.load_default()

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

# ---------- TTS (Edge + word timings; fallback gTTS) ----------
async def edge_tts_synthesize_async(text: str, outpath: str, voice: str) -> List[Tuple[str, float, float]]:
    """
    Return: [(word, start_s, end_s), ...]  (kelime sınırları)
    """
    import edge_tts, aiofiles
    timings: List[Tuple[str, float, float]] = []
    communicator = edge_tts.Communicate(text, voice=voice)  # boundary için stream'e bakacağız
    cur_word_start = None
    cur_word = None
    async with aiofiles.open(outpath, "wb") as f:
        async for chunk in communicator.stream():
            if chunk["type"] == "audio":
                await f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary","Boundary"):
                # edge-tts bazen 'offset' ya da 'audio_offset' verir; 100-ns → saniye
                off = chunk.get("offset") or chunk.get("audio_offset")
                dur = chunk.get("duration", 0)
                txt = (chunk.get("text") or "").strip()
                try:
                    start_s = (off or 0) / 10_000_000.0
                    dur_s = (dur or 0) / 10_000_000.0
                except Exception:
                    start_s, dur_s = 0.0, 0.0
                if txt:
                    timings.append((txt, start_s, start_s + max(dur_s, 0.06)))
    # filtrele: çok kısa kelimeleri de kısacık gösterelim
    out = []
    for w, s, e in timings:
        if e <= s: e = s + 0.08
        out.append((w, s, e))
    return out

def tts_to_file_with_timings(text: str, outpath: str, lang: str = "en", voice: Optional[str] = None) -> List[Tuple[str,float,float]]:
    # Edge-TTS dene
    if voice:
        try:
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            timings = asyncio.get_event_loop().run_until_complete(edge_tts_synthesize_async(text, outpath, voice))
            if timings:
                return timings
        except Exception:
            pass
    # Fallback gTTS (timings yok → boş liste)
    gTTS(text=text, lang=lang, tld="com").save(outpath)
    return []

# ---------- Ambient (procedural) ----------
def generate_ambient_wav(path: str, duration_s: float, sr: int = 44100):
    t = np.linspace(0, duration_s, int(sr*duration_s), endpoint=False)
    # 3 düşük frekanslı drone + çok düşük genlik
    freqs = [110.0, 220.0, 275.0]
    wave = sum(np.sin(2*np.pi*f*t + ph) for f, ph in zip(freqs, [0.0, 1.3, 2.1])) / 3.0
    # hafif tremolo
    trem = 0.5*(1 + np.sin(2*np.pi*0.2*t))
    audio = (wave * trem * 0.08).astype(np.float32)  # düşük volüm
    # fade in/out
    fade_len = int(sr*1.0)
    env = np.ones_like(audio)
    env[:fade_len] = np.linspace(0, 1, fade_len)
    env[-fade_len:] = np.linspace(1, 0, fade_len)
    audio *= env
    # yaz
    import soundfile as sf
    sf.write(path, audio, sr)

def ensure_ambient(duration_s: float) -> Optional[str]:
    # Eğer assets/ambience.mp3 varsa onu kullan
    custom = ASSETS_DIR / "ambience.mp3"
    if custom.exists():
        return str(custom)
    # yoksa procedural üret (WAV)
    try:
        out = "ambient_autoyuson.wav"
        # soundfile yoksa kurulu değilse MoviePy'nin AudioClip ile yazması zahmetli; bu yüzden requirements'a soundfile ekledik
        generate_ambient_wav(out, duration_s)
        return out
    except Exception:
        return None

# ---------- Video yardımcıları ----------
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

def ken_burns_clip(img: Image.Image, duration: float, zoom_start=1.06, zoom_end=1.12, pan_dx=0, pan_dy=0):
    """Tek görsel için garantili tam-süre Ken Burns."""
    path = "kb_tmp.jpg"; fit_center_crop(img).save(path, quality=92)
    base = ImageClip(path).set_duration(duration)
    # zamanla ölçek ve küçük pan
    def scaler(t):
        k = zoom_start + (zoom_end - zoom_start) * (t / max(duration, 0.001))
        return k
    clip = base.resize(lambda t: scaler(t)).set_position(("center","center"))
    return clip

def build_broll(bg_images: List[Image.Image], duration: float):
    if not bg_images:
        bg_images = [Image.new("RGB",(W,H),(20,20,24))]
    n = max(1, len(bg_images))
    if n == 1:
        return ken_burns_clip(bg_images[0], duration)
    # birden çok görsel → hepsine kısmi zoom ve crossfade bindirme
    per = duration / n
    per = max(3.5, min(8.0, per))
    fc = 0.6  # crossfade
    layers = []
    t = 0.0
    for i, im in enumerate(bg_images):
        clip = ken_burns_clip(im, per, zoom_start=1.05, zoom_end=1.12)
        layers.append(clip.set_start(t).crossfadein(fc).crossfadeout(fc))
        t += per - fc
    comp = CompositeVideoClip(layers).set_duration(duration)
    return comp

# ---------- Sub timing ----------
def split_words(text: str) -> List[str]:
    return [w for w in re.findall(r"\S+", text)]

def chunk_words(words: List[str], max_per=3) -> List[List[str]]:
    out, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) >= max_per:
            out.append(buf); buf=[]
    if buf: out.append(buf)
    return out

def words_to_chunks_with_timings(words: List[str], duration: float) -> List[Tuple[str,float,float]]:
    # Kelime başı ~0.18–0.28s gibi hızlı bir akış
    n = max(1, len(words))
    avg = max(0.18, min(0.28, duration / (n+4)))
    t = 0.8
    out = []
    for w in words:
        start = t; end = min(duration-0.2, t + avg)
        out.append((w, start, end))
        t = end - 0.02  # ufak bindirme
        if t >= duration-0.2: break
    return out

def group_or_from_boundaries(boundaries: List[Tuple[str,float,float]], duration: float) -> List[Tuple[str,float,float]]:
    if not boundaries:
        return []
    # doğrudan kelime altyazısı (çok hızlı olabilir). Okunurluk için 2'şer veya 3'er gruplar halinde topla.
    chunks = []
    buf: List[Tuple[str,float,float]] = []
    for w, s, e in boundaries:
        if not buf: buf.append((w,s,e))
        else:
            # grupla; toplam süre çok uzamasın
            if len(buf) < 3 and (e - buf[0][1]) < 1.1:
                buf.append((w,s,e))
            else:
                text = " ".join(x[0] for x in buf)
                chunks.append((text, buf[0][1], buf[-1][2]))
                buf = [(w,s,e)]
    if buf:
        text = " ".join(x[0] for x in buf)
        chunks.append((text, buf[0][1], buf[-1][2]))
    # sınırlar
    norm = []
    for text, s, e in chunks:
        s = max(0.6, min(s, duration-0.3))
        e = max(s+0.1, min(e, duration-0.2))
        norm.append((text, s, e))
    return norm

# ---------- Render ----------
def render_video(title: str, narration: str, audio_path: str,
                 word_boundaries: List[Tuple[str,float,float]],
                 bg_images: List[Image.Image], duration_hint: int,
                 wiki_title_url: str) -> RenderResult:
    ensure_fonts()

    narration_audio = AudioFileClip(audio_path)
    duration = max(min(max(duration_hint, 45), 75), narration_audio.duration + 0.6)

    # B-roll
    broll = build_broll(bg_images, duration)

    # Başlık (daha küçük, üstte)
    title_png = "title.png"
    save_text_png(title[:64], title_png, width=W-140, fontsize=64, bold=True)
    title_clip = ImageClip(title_png).set_duration(min(5.0, duration*0.2)).set_position(("center", 64)).fadein(0.3).fadeout(0.3)

    # Altyazılar (küçük, kelime/mini-chunk senkron)
    subs = []
    if word_boundaries:
        chunks = group_or_from_boundaries(word_boundaries, narration_audio.duration)
    else:
        words = split_words(narration)
        chunks_raw = chunk_words(words, max_per=3)
        # süreyi toplam kelimeye göre dağıt
        chunks = []
        word_seq = [w for grp in chunks_raw for w in grp]
        est = words_to_chunks_with_timings(word_seq, narration_audio.duration)
        # est ile chunk’ları eşleştir
        i = 0
        for grp in chunks_raw:
            if i >= len(est): break
            s = est[i][1]; e = est[min(i+len(grp)-1, len(est)-1)][2]
            chunks.append((" ".join(grp), s, e))
            i += len(grp)

    for idx, (text, s, e) in enumerate(chunks):
        fn = f"sub_{idx:03d}.png"
        save_text_png(text, fn, width=W-160, fontsize=42, bold=False)  # daha küçük
        clip = ImageClip(fn).set_start(s).set_duration(max(0.12, e - s)).set_position(("center", H-360)).fadein(0.08).fadeout(0.08)
        subs.append(clip)

    # Ambiyans
    amb_path = ensure_ambient(duration)
    audio_layers = [narration_audio.set_start(0).fx(afx.audio_normalize)]
    if amb_path:
        amb = AudioFileClip(amb_path).volumex(0.18).audio_fadein(0.8).audio_fadeout(0.8)
        amb = amb.set_duration(duration)
        audio_layers.append(amb)

    comp_audio = CompositeAudioClip(audio_layers)

    comp = CompositeVideoClip([broll, title_clip, *subs]).set_audio(comp_audio).set_duration(duration)

    out_name = re.sub(r"[^-\w\s.,()]+","",title).strip().replace(" ","_")[:80] or "autoyuson_video"
    out_path = f"{out_name}.mp4"
    comp.write_videofile(out_path, fps=30, codec="libx264", audio_codec="aac",
                         temp_audiofile="temp-audio.m4a", remove_temp=True, threads=4, verbose=False, logger=None)

    wiki_url = f"{WIKI}/wiki/{urllib.parse.quote(wiki_title_url)}"
    description = (f"{title}\n\nShort factual video. No speculation, no fiction.\n\n"
                   f"Source:\n- Wikipedia: {wiki_url}"
                   "\n\n#history #space #culture #facts #shorts")
    return RenderResult(video_path=out_path, title=title, description=description, tags=DEFAULT_TAGS)

# ---------- OAuth helpers / YouTube ----------
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
    narration, short_title, sentences = craft_script(title, summary)
    if not is_factual_enough(narration):
        pass

    # Görselleri indir; en az 3’e tamamlamak için tekrar kullan
    bg_images: List[Image.Image] = []
    for u in image_urls:
        im = download_image(u)
        if im: bg_images.append(im)
    if not bg_images:
        bg_images = [Image.new("RGB",(W,H),(20,20,24))]
    while len(bg_images) < 3:
        bg_images.append(bg_images[-1])

    # TTS + word timings
    audio_path = "narration_autoyuson.mp3"
    boundaries = tts_to_file_with_timings(narration, audio_path, LANG, voice=EDGE_TTS_VOICE)

    # Render
    result = render_video(short_title, narration, audio_path, boundaries, bg_images, DURATION_TARGET, wiki_title_url=title.replace(" ","_"))

    # Upload
    print("[AUTOYUSON] Uploading...")
    vid = youtube_upload(result.video_path, result.title, result.description, result.tags)
    print(f"[AUTOYUSON] Uploaded: https://youtube.com/watch?v={vid}")

    seen.append(title); save_seen(seen)
    print("[AUTOYUSON] Seen topics updated.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
