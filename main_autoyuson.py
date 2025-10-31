"""
AUTOYUSON — Factual Shorts (History / Space / Culture) for Yusonvasya

İyileştirmeler:
- Metin: Hook → Dense Facts → Closure (factual, ilgi çekici)
- Ses: Edge-TTS + SSML (narration-professional, rate +4%, pitch -2%) → fallback gTTS
- Altyazı: Edge WordBoundary ile mini-chunk senkron; yoksa akıcı tahmin
- Altyazı okunabilirliği: yarı saydam arka plan şeridi
- Görsel: Tek görselde tam süre Ken Burns; çok görselde crossfade
- Ambiyans: assets/ambience.mp3 varsa kullan; yoksa procedural drone (çok düşük volüm)

ENV (GitHub Secrets veya local):
  YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN
  # Yedek:
  TOKEN_JSON_BASE64   (token.json base64; içinde refresh_token + client_id + client_secret)

Opsiyonel:
  LANGUAGE=en
  VIDEO_DURATION=55
  TITLE_PREFIX=
  TOPICS_FILE=topics.txt
  EDGE_TTS_VOICE=en-US-JennyNeural  # daha doğal
"""

import os, io, re, json, base64, random, textwrap, pathlib, sys, asyncio, urllib.parse
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
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")  # daha doğal hissiyat

DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
SEEN_PATH = DATA_DIR / "seen_topics.json"
FONTS_DIR = pathlib.Path("fonts"); FONTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR = pathlib.Path("assets"); ASSETS_DIR.mkdir(exist_ok=True)

W, H = 1080, 1920
WIKI = "https://en.wikipedia.org"
REST = f"{WIKI}/api/rest_v1"
API  = f"{WIKI}/w/api.php"
UA = {"User-Agent": "autoyuson-bot/1.2 (+github)"}

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
            return title, extract, page_url, imgs[:8]
    raise RuntimeError(f"Wiki fetch failed for topic: {topic}")

def download_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, headers=UA, timeout=30); r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

# ---------- Script (hook → facts → close) ----------
def craft_script(title: str, summary: str) -> Tuple[str, str, List[str]]:
    # kanca için yıl/ölçü yakala
    years = re.findall(r"(1[0-9]{3}|20[0-9]{2})", summary)
    nums  = re.findall(r"\b([0-9]+(?:\.[0-9]+)?)(?:\s?(?:km|m|kg|years?|%|million|billion))?\b", summary, flags=re.I)
    hook_bits = []
    if years: hook_bits.append(years[0])
    if nums:  hook_bits.append(nums[0])
    hook = f"{title}: {', '.join(hook_bits)} — evidence that changed what we thought we knew." if hook_bits else f"{title}: evidence that changed what we thought we knew."

    # temiz metin
    txt = re.sub(r"\s+", " ", summary).strip()
    words = txt.split()
    body = " ".join(words[:120]) if len(words) > 120 else (txt if len(words) >= 80 else (txt + " " + " ".join((words+words)[:80])))

    close = "In history, what endures is what we can verify — and what it reveals."
    narration = f"{hook} {body} {close}"
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

def draw_text_with_bg(width: int, lines: List[str], fontsize: int, bold: bool=False,
                      fg=(255,255,255), bg=(0,0,0,140), pad=14, radius=18) -> Image.Image:
    font = get_font(size=fontsize, bold=bold)
    draw = ImageDraw.Draw(Image.new("RGB",(10,10)))
    line_w = [draw.textsize(ln, font=font)[0] for ln in lines]
    ascent, descent = font.getmetrics()
    lh = ascent + descent + 6
    box_w = min(width, max(line_w) + pad*2)
    box_h = lh*len(lines) + pad*2
    img = Image.new("RGBA", (width, box_h), (0,0,0,0))
    # rounded rect
    rect = Image.new("RGBA", (box_w, box_h), bg)
    img.alpha_composite(rect, ((width - box_w)//2, 0))
    # text
    d = ImageDraw.Draw(img)
    y = pad
    for ln in lines:
        w,_ = d.textsize(ln, font=font)
        x = (width - w)//2
        d.text((x, y), ln, font=font, fill=fg)
        y += lh
    return img

def save_text_box(text: str, out_path: str, width: int, fontsize: int, bold=False):
    # satır kır
    font = get_font(size=fontsize, bold=bold)
    d = ImageDraw.Draw(Image.new("RGB",(10,10)))
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        wpx,_ = d.textsize(test, font=font)
        if wpx > width - 40:
            if cur: lines.append(cur)
            cur = w
        else:
            cur = test
    if cur: lines.append(cur)
    img = draw_text_with_bg(width, lines, fontsize, bold)
    img.save(out_path, "PNG")

# ---------- TTS (Edge + SSML + WordBoundary; fallback gTTS) ----------
SSML_TMPL = """<speak version="1.0" xml:lang="en-US">
  <voice name="{voice}">
    <mstts:express-as style="narration-professional">
      <prosody rate="+4%" pitch="-2%">
        {payload}
      </prosody>
    </mstts:express-as>
  </voice>
</speak>"""

def sentences_to_ssml(sentences: List[str]) -> str:
    # Her cümle arası 200ms
    parts = []
    for i, s in enumerate(sentences):
        s = s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        parts.append(f"<p>{s}</p>")
        if i != len(sentences)-1:
            parts.append('<break time="200ms"/>')
    return "".join(parts)

async def edge_tts_synthesize_ssml_async(sentences: List[str], outpath: str, voice: str) -> List[Tuple[str,float,float]]:
    import edge_tts, aiofiles
    ssml = SSML_TMPL.format(voice=voice, payload=sentences_to_ssml(sentences))
    communicator = edge_tts.Communicate(ssml, voice=voice)
    timings: List[Tuple[str,float,float]] = []
    async with aiofiles.open(outpath, "wb") as f:
        async for chunk in communicator.stream():
            if chunk["type"] == "audio":
                await f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary","Boundary"):
                off = chunk.get("offset") or chunk.get("audio_offset") or 0
                dur = chunk.get("duration", 0)
                txt = (chunk.get("text") or "").strip()
                try:
                    s = off / 10_000_000.0
                    d = (dur or 0) / 10_000_000.0
                except Exception:
                    s, d = 0.0, 0.0
                if txt:
                    timings.append((txt, s, s + max(d, 0.06)))
    # normalize
    out = []
    for w, s, e in timings:
        if e <= s: e = s + 0.08
        out.append((w, s, e))
    return out

def tts_with_timings(sentences: List[str], narration_text: str, outpath: str, lang: str, voice: Optional[str]) -> List[Tuple[str,float,float]]:
    # Edge-SSML dene
    if voice:
        try:
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            timings = asyncio.get_event_loop().run_until_complete(edge_tts_synthesize_ssml_async(sentences, outpath, voice))
            if timings:
                return timings
        except Exception:
            pass
    # Fallback: gTTS (timings yok)
    gTTS(text=narration_text, lang=lang, tld="com").save(outpath)
    return []

# ---------- Ambient (procedural) ----------
def generate_ambient_wav(path: str, duration_s: float, sr: int = 44100):
    t = np.linspace(0, duration_s, int(sr*duration_s), endpoint=False)
    freqs = [110.0, 220.0, 275.0]
    phase = [0.0, 1.1, 2.0]
    wave = sum(np.sin(2*np.pi*f*t + p) for f, p in zip(freqs, phase)) / 3.0
    trem = 0.5*(1 + np.sin(2*np.pi*0.2*t))
    audio = (wave * trem * 0.08).astype(np.float32)
    # fade
    fade = int(sr*0.8)
    env = np.ones_like(audio)
    env[:fade] = np.linspace(0,1,fade)
    env[-fade:] = np.linspace(1,0,fade)
    audio *= env
    import soundfile as sf
    sf.write(path, audio, sr)

def ensure_ambient(duration_s: float) -> Optional[str]:
    custom = ASSETS_DIR / "ambience.mp3"
    if custom.exists():
        return str(custom)
    try:
        out = "ambient_autoyuson.wav"
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

def ken_burns_clip(img: Image.Image, duration: float, zoom_start=1.05, zoom_end=1.12):
    path = "kb_tmp.jpg"; fit_center_crop(img).save(path, quality=92)
    base = ImageClip(path).set_duration(duration)
    def scaler(t):
        return zoom_start + (zoom_end - zoom_start) * (t / max(duration, 0.001))
    return base.resize(lambda t: scaler(t)).set_position(("center","center"))

def build_broll(bg_images: List[Image.Image], duration: float):
    if not bg_images:
        bg_images = [Image.new("RGB",(W,H),(20,20,24))]
    n = max(1, len(bg_images))
    if n == 1:
        return ken_burns_clip(bg_images[0], duration)
    per = max(3.5, min(8.0, duration / n))
    fc = 0.6
    layers, t = [], 0.0
    for im in bg_images:
        clip = ken_burns_clip(im, per)
        layers.append(clip.set_start(t).crossfadein(fc).crossfadeout(fc))
        t += per - fc
    return CompositeVideoClip(layers).set_duration(duration)

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
    n = max(1, len(words))
    avg = max(0.18, min(0.26, duration / (n + 6)))
    t = 0.7
    out = []
    for w in words:
        start = t; end = min(duration-0.2, t + avg)
        out.append((w, start, end))
        t = end - 0.02
        if t >= duration-0.2: break
    return out

def group_from_boundaries(boundaries: List[Tuple[str,float,float]], duration: float) -> List[Tuple[str,float,float]]:
    if not boundaries:
        return []
    chunks, buf = [], []
    for w, s, e in boundaries:
        if not buf:
            buf = [(w,s,e)]
        else:
            if len(buf) < 3 and (e - buf[0][1]) < 1.1:
                buf.append((w,s,e))
            else:
                text = " ".join(x[0] for x in buf)
                chunks.append((text, buf[0][1], buf[-1][2]))
                buf = [(w,s,e)]
    if buf:
        text = " ".join(x[0] for x in buf)
        chunks.append((text, buf[0][1], buf[-1][2]))
    norm = []
    for text, s, e in chunks:
        s = max(0.6, min(s, duration-0.3))
        e = max(s+0.12, min(e, duration-0.2))
        norm.append((text, s, e))
    return norm

# ---------- Render ----------
def render_video(title: str, narration: str, sentences: List[str], audio_path: str,
                 word_boundaries: List[Tuple[str,float,float]],
                 bg_images: List[Image.Image], duration_hint: int, wiki_title_url: str) -> RenderResult:
    ensure_fonts()

    narration_audio = AudioFileClip(audio_path)
    duration = max(min(max(duration_hint, 48), 75), narration_audio.duration + 0.6)

    broll = build_broll(bg_images, duration)

    # Başlık
    title_png = "title.png"
    save_text_box(title[:64], title_png, width=W-140, fontsize=60, bold=True)
    title_clip = ImageClip(title_png).set_duration(min(5.0, duration*0.22)).set_position(("center", 72)).fadein(0.3).fadeout(0.3)

    # Altyazılar
    subs = []
    if word_boundaries:
        chunks = group_from_boundaries(word_boundaries, narration_audio.duration)
    else:
        words = split_words(narration)
        chunks_raw = chunk_words(words, max_per=3)
        est = words_to_chunks_with_timings([w for g in chunks_raw for w in g], narration_audio.duration)
        i = 0; chunks = []
        for grp in chunks_raw:
            if i >= len(est): break
            s = est[i][1]; e = est[min(i+len(grp)-1, len(est)-1)][2]
            chunks.append((" ".join(grp), s, e))
            i += len(grp)

    for idx, (text, s, e) in enumerate(chunks):
        fn = f"sub_{idx:03d}.png"
        save_text_box(text, fn, width=W-200, fontsize=38, bold=False)  # küçük + arka planlı
        clip = ImageClip(fn).set_start(s).set_duration(max(0.12, e - s)).set_position(("center", H-340)).fadein(0.07).fadeout(0.07)
        subs.append(clip)

    # Ambiyans
    amb_path = ensure_ambient(duration)
    audio_layers = [narration_audio.set_start(0).fx(afx.audio_normalize)]
    if amb_path:
        amb = AudioFileClip(amb_path).volumex(0.14).audio_fadein(0.7).audio_fadeout(0.8).set_duration(duration)
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

# ---------- OAuth / Upload ----------
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
    # Topic seç
    seen = load_seen()
    topic = pick_topic(seen)
    print(f"[AUTOYUSON] Topic: {topic}")

    # Wiki verisi
    title, summary, page_url, image_urls = wiki_fetch(topic)

    # Metin
    narration, short_title, sentences = craft_script(title, summary)
    if not is_factual_enough(narration):
        pass

    # Görseller
    bg_images: List[Image.Image] = []
    for u in image_urls:
        im = download_image(u)
        if im: bg_images.append(im)
    if not bg_images:
        bg_images = [Image.new("RGB",(W,H),(20,20,24))]
    while len(bg_images) < 3:
        bg_images.append(bg_images[-1])

    # TTS + timings
    audio_path = "narration_autoyuson.mp3"
    word_boundaries = tts_with_timings(sentences, narration, audio_path, LANG, EDGE_TTS_VOICE)

    # Render
    result = render_video(short_title, narration, sentences, audio_path,
                          word_boundaries, bg_images, DURATION_TARGET,
                          wiki_title_url=title.replace(" ","_"))

    # Upload
    print("[AUTOYUSON] Uploading...")
    vid = youtube_upload(result.video_path, result.title, result.description, result.tags)
    print(f"[AUTOYUSON] Uploaded: https://youtube.com/watch?v={vid}")

    seen.append(title); save_seen(seen)
    print("[AUTOYUSON] Seen topics updated.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
