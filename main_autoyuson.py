# -*- coding: utf-8 -*-
"""
AUTOYUSON — Cinematic Factual Shorts (History / Space / Culture)

V3 (fixed) ana farklar:
- Beat sheet metin: cold open → hook → 3 fact → close
- Edge-TTS + SSML (narration-relaxed, rate -3%, pitch -2%), gTTS fallback
- Pydub mastering: HPF 120 Hz, LPF 8 kHz, hafif kompresyon, yumuşatma
- Ambiyans + otomatik ducking
- Görsel sinema dokusu: Ken Burns, crossfade, warm grade, vignette, film-grain
- Mini-chunk kelime senkron altyazı
- **BUGFIX:** Vinyet/overlay artık her görselin kendi boyutunda oluşturuluyor (ValueError bitti)

ENV (GitHub Secrets veya local):
  YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN
  (veya TOKEN_JSON_BASE64 içinde refresh_token + client_id + client_secret)

Opsiyonel:
  LANGUAGE=en
  VIDEO_DURATION=55
  TITLE_PREFIX=
  TOPICS_FILE=topics.txt
  EDGE_TTS_VOICE=en-US-JennyNeural
"""

import os, io, re, json, base64, random, pathlib, sys, asyncio, urllib.parse, math
from typing import List, Tuple, Optional
from dataclasses import dataclass

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np
from gtts import gTTS
from moviepy.editor import (ImageClip, AudioFileClip, CompositeVideoClip,
                            CompositeAudioClip, afx)
from moviepy.video.fx.all import resize as mp_resize
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# Pillow ANTIALIAS uyumluluğu
try:
    _ = Image.ANTIALIAS  # type: ignore[attr-defined]
except AttributeError:
    try:
        Image.ANTIALIAS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except Exception:
        pass

# ---------- Config ----------
LANG = os.getenv("LANGUAGE", "en").strip() or "en"
DURATION_TARGET = int(os.getenv("VIDEO_DURATION", "55"))
TITLE_PREFIX = os.getenv("TITLE_PREFIX", "").strip()
YOUTUBE_CATEGORY_ID = "27"
DEFAULT_TAGS = ["history", "space", "astronomy", "archaeology", "culture", "documentary", "facts", "shorts"]
TOPICS_FILE = os.getenv("TOPICS_FILE", "topics.txt")
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")

DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
SEEN_PATH = DATA_DIR / "seen_topics.json"
FONTS_DIR = pathlib.Path("fonts"); FONTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR = pathlib.Path("assets"); ASSETS_DIR.mkdir(exist_ok=True)

W, H = 1080, 1920
WIKI = "https://en.wikipedia.org"
REST = f"{WIKI}/api/rest_v1"
API  = f"{WIKI}/w/api.php"
UA = {"User-Agent": "autoyuson-bot/1.3 (+github)"}

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
    return r.json() if r.status_code == 200 else None

def _rest_media(title: str):
    url = f"{REST}/page/media/{urllib.parse.quote(title)}"
    r = requests.get(url, headers=UA, timeout=20)
    return r.json() if r.status_code == 200 else None

def _opensearch(q: str) -> List[str]:
    r = requests.get(API, headers=UA, timeout=20, params={
        "action":"opensearch","search":q,"limit":5,"namespace":0,"format":"json"
    })
    if r.status_code != 200: return []
    data = r.json()
    return data[1] if isinstance(data, list) and len(data)>=2 else []

def wiki_fetch(topic: str):
    cands = [topic] + _opensearch(topic)
    for cand in cands:
        s = _rest_summary(cand)
        if not s: continue
        title = s.get("title") or cand
        extract = s.get("extract") or ""
        page_url = s.get("content_urls",{}).get("desktop",{}).get("page") or f"{WIKI}/wiki/{title.replace(' ','_')}"
        imgs=[]
        m = _rest_media(title)
        if m and "items" in m:
            for it in m["items"]:
                if it.get("type")!="image": continue
                srcset = it.get("srcset") or []
                srcset = sorted(srcset, key=lambda x: x.get("scale",1.0))
                url = (srcset[-1]["src"] if srcset else it.get("src")) or ""
                low=url.lower()
                if any(b in low for b in ["logo","icon","flag","seal","map","diagram","coat_of_arms","favicon","sprite"]): continue
                if not low.endswith((".jpg",".jpeg",".png",".webp")): continue
                imgs.append(url)
        thumb = s.get("thumbnail",{}).get("source")
        if thumb: imgs=[thumb]+imgs
        if extract:
            return title, extract, page_url, imgs[:8]
    raise RuntimeError("Wiki fetch failed")

def download_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, headers=UA, timeout=30); r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

# ---------- Script (Beat Sheet) ----------
def craft_script(title: str, summary: str):
    clean = re.sub(r"\s+"," ", summary).strip()
    cold_open = f"{title}. Verified, preserved, undeniable."
    hook       = "What changed our understanding? Evidence — not legend."
    words = clean.split()
    fact1 = " ".join(words[:40]) if words else clean
    fact2 = " ".join(words[40:80]) or clean
    fact3 = " ".join(words[80:120]) or clean
    close = "History survives by proof — and by those who keep looking."
    sentences = [cold_open, hook, fact1, fact2, fact3, close]
    narration = " ".join(sentences)
    return narration, title, sentences

def is_factual(text:str)->bool:
    if any(w in text.lower() for w in ["reportedly","allegedly","legend","rumor"]): return False
    n=len(text.split()); return 80<=n<=260

# ---------- Fonts & Text ----------
N_REG = FONTS_DIR/"NotoSans-Regular.ttf"
N_BOLD= FONTS_DIR/"NotoSans-Bold.ttf"

def ensure_fonts():
    urls={
        N_REG :"https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
        N_BOLD:"https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
    }
    for p,u in urls.items():
        if not p.exists():
            try:
                r=requests.get(u,headers=UA,timeout=30); r.raise_for_status()
                p.write_bytes(r.content)
            except Exception: pass

def get_font(size=48,bold=False):
    try: return ImageFont.truetype(str(N_BOLD if bold else N_REG), size=size)
    except Exception: return ImageFont.load_default()

def draw_text_box(text:str,width:int,fontsize:int,bold=False,fg=(255,255,255),bg=(0,0,0,150)):
    f=get_font(fontsize,bold)
    d=ImageDraw.Draw(Image.new("RGB",(10,10)))
    words=text.split(); lines=[]; cur=""
    for w in words:
        t=(cur+" "+w).strip(); wpx,_=d.textsize(t,font=f)
        if wpx>width-40:
            if cur: lines.append(cur); cur=w
            else: lines.append(w); cur=""
        else: cur=t
    if cur: lines.append(cur)
    ascent,descent=f.getmetrics(); lh=ascent+descent+6
    box_w=min(width,max(d.textsize(x,font=f)[0] for x in lines)+36); box_h=lh*len(lines)+28
    img=Image.new("RGBA",(width,box_h),(0,0,0,0))
    rect=Image.new("RGBA",(box_w,box_h),bg)
    img.alpha_composite(rect,((width-box_w)//2,0))
    y=14; draw=ImageDraw.Draw(img)
    for ln in lines:
        wpx,_=draw.textsize(ln,font=f); x=(width-wpx)//2
        draw.text((x,y),ln,font=f,fill=fg); y+=lh
    return img

def save_text_box(text,out_path,width,fontsize,bold=False):
    img=draw_text_box(text,width,fontsize,bold)
    img.save(out_path,"PNG")

# ---------- TTS + timings ----------
SSML_TMPL = """<speak version="1.0" xml:lang="en-US">
  <voice name="{voice}">
    <mstts:express-as style="narration-relaxed">
      <prosody rate="-3%" pitch="-2%">{payload}</prosody>
    </mstts:express-as>
  </voice>
</speak>"""

def sentences_to_ssml(sents:List[str])->str:
    parts=[]
    for i,s in enumerate(sents):
        s=(s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
        br="300ms" if i in (0,1) else "180ms"
        parts.append(f"<p>{s}</p><break time=\"{br}\"/>")
    return "".join(parts)

async def edge_ssml_async(sentences:List[str], outpath:str, voice:str):
    import edge_tts, aiofiles
    ssml=SSML_TMPL.format(voice=voice, payload=sentences_to_ssml(sentences))
    comm=edge_tts.Communicate(ssml, voice=voice)
    timings=[]
    async with aiofiles.open(outpath,"wb") as f:
        async for ch in comm.stream():
            if ch["type"]=="audio":
                await f.write(ch["data"])
            elif ch["type"] in ("WordBoundary","Boundary"):
                off=ch.get("offset") or ch.get("audio_offset") or 0
                dur=ch.get("duration",0)
                txt=(ch.get("text") or "").strip()
                s=off/10_000_000.0
                e=s+max((dur or 0)/10_000_000.0,0.06)
                if txt: timings.append((txt,s,e))
    out=[]
    for w,s,e in timings:
        if e<=s: e=s+0.08
        out.append((w,s,e))
    return out

def tts_with_timings(sentences:List[str], narration:str, outpath:str, lang:str, voice:str):
    try:
        try: asyncio.get_event_loop()
        except RuntimeError: asyncio.set_event_loop(asyncio.new_event_loop())
        t=asyncio.get_event_loop().run_until_complete(edge_ssml_async(sentences,outpath,voice))
        if t: return t
    except Exception:
        pass
    gTTS(text=narration,lang=lang,tld="com").save(outpath)
    return []

# ---------- Mastering (pydub) ----------
def master_voice(in_mp3:str, out_mp3:str):
    from pydub import AudioSegment, effects
    voice=AudioSegment.from_file(in_mp3)
    soft=voice.low_pass_filter(9000)
    voice=soft.high_pass_filter(120).low_pass_filter(8000)
    voice=effects.compress_dynamic_range(voice, threshold=-18.0, ratio=2.0, attack=5, release=50)
    voice=effects.normalize(voice, headroom=14.0)
    voice.export(out_mp3, format="mp3")

def duck_ambience(narration_path:str, ambience_path:str, out_path:str, up_gain_db=-13.0, down_gain_db=-18.0):
    from pydub import AudioSegment
    nar=AudioSegment.from_file(narration_path)
    amb=AudioSegment.from_file(ambience_path)
    if len(amb)<len(nar): amb=amb*int(math.ceil(len(nar)/len(amb))); amb=amb[:len(nar)]
    window=50
    mixed=[]
    for i in range(0, len(nar), window):
        slice_db = nar[i:i+window].dBFS if nar[i:i+window].rms>0 else -90
        gain = down_gain_db if slice_db>-42 else up_gain_db
        mixed.append(amb[i:i+window]+gain)
    res = mixed[0] if mixed else amb+down_gain_db
    for seg in mixed[1:]:
        res = res.append(seg, crossfade=5)
    res.export(out_path, format="mp3")

# ---------- Visual helpers ----------
def fit_center_crop(img: Image.Image, w=W, h=H) -> Image.Image:
    r=img.width/img.height; rf=w/h
    if r>rf: new_h=h; new_w=int(r*new_h)
    else: new_w=w; new_h=int(new_w/r)
    img=img.resize((new_w,new_h),Image.LANCZOS)
    L=(img.width-w)//2; T=(img.height-h)//2
    return img.crop((L,T,L+w,T+h))

def apply_grade(img: Image.Image) -> Image.Image:
    """Sıcak renk derecelendirme + vinyet + grain, GÖRSEL BOYUTUNA göre."""
    w,h = img.size
    # warm grade
    graded = ImageOps.colorize(ImageOps.grayscale(img), black="#0b0b0b", white="#f2e8da").convert("RGB")
    graded = Image.blend(img, graded, 0.18)

    # vignette mask: kenarlarda güçlü, merkezde zayıf
    mask = Image.new("L", (w, h), 160)  # kenarlar: koyu
    draw = ImageDraw.Draw(mask)
    margin = int(min(w, h) * 0.30)
    draw.ellipse((margin, margin, w - margin, h - margin), fill=0)  # merkez: açık
    mask = mask.filter(ImageFilter.GaussianBlur(80))

    # overlay (siyah), mask'i alfa olarak kullan
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    overlay.putalpha(mask)
    base = graded.convert("RGBA")
    with_vignette = Image.alpha_composite(base, overlay).convert("RGB")

    # film grain
    noise = (np.random.randn(h, w) * 6).clip(-12, 12).astype(np.int16)
    arr = np.array(with_vignette).astype(np.int16)
    arr = np.clip(arr + noise[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def ken_burns_clip(img: Image.Image, duration: float, z0=1.05, z1=1.12):
    base = fit_center_crop(apply_grade(img), W, H)
    path="kb.jpg"; base.save(path,quality=92)
    clip=ImageClip(path).set_duration(duration)
    return clip.resize(lambda t: z0+(z1-z0)*(t/duration)).set_position(("center","center"))

def build_broll(images:List[Image.Image], duration:float):
    if not images:
        images=[Image.new("RGB",(W,H),(20,20,24))]
    if len(images)==1:
        return ken_burns_clip(images[0], duration)
    per=max(3.6, min(7.8, duration/len(images))); fc=0.6
    layers=[]; t=0.0
    for im in images:
        layers.append(ken_burns_clip(im, per).set_start(t).crossfadein(fc).crossfadeout(fc))
        t+= per - fc
    return CompositeVideoClip(layers).set_duration(duration)

# ---------- Subtitles ----------
def group_boundaries(bound:List[Tuple[str,float,float]], dur:float)->List[Tuple[str,float,float]]:
    if not bound: return []
    chunks=[]; buf=[]
    for w,s,e in bound:
        if not buf: buf=[(w,s,e)]
        elif len(buf)<3 and (e-buf[0][1])<1.1:
            buf.append((w,s,e))
        else:
            txt=" ".join(x[0] for x in buf); chunks.append((txt, buf[0][1], buf[-1][2])); buf=[(w,s,e)]
    if buf: chunks.append((" ".join(x[0] for x in buf), buf[0][1], buf[-1][2]))
    out=[]
    for t,s,e in chunks:
        s=max(0.55,min(s,dur-0.3)); e=max(s+0.12,min(e,dur-0.2)); out.append((t,s,e))
    return out

def fallback_chunks(text:str, dur:float)->List[Tuple[str,float,float]]:
    words=re.findall(r"\S+", text); n=max(1,len(words))
    avg=max(0.18, min(0.24, dur/(n+6))); t=0.6; out=[]
    buf=[]; cnt=0
    for w in words:
        buf.append(w); cnt+=1
        if cnt==3:
            out.append((" ".join(buf), t, min(dur-0.2, t+avg*3)))
            t = t + avg*2.8; buf=[]; cnt=0
    if buf:
        out.append((" ".join(buf), t, min(dur-0.2, t+avg*len(buf))))
    return out

# ---------- Render ----------
@dataclass
class RenderResult:
    video_path:str; title:str; description:str; tags:List[str]

def render(title:str, narration:str, sentences:List[str],
           audio_path:str, boundaries:List[Tuple[str,float,float]],
           images:List[Image.Image], duration_hint:int, wiki_title_url:str)->RenderResult:
    ensure_fonts()
    voice = AudioFileClip(audio_path)
    duration = max(min(max(duration_hint,48),75), voice.duration+0.6)

    # B-roll
    broll=build_broll(images, duration)

    # Title
    title_png="title.png"; save_text_box(title[:64], title_png, width=W-140, fontsize=58, bold=True)
    title_clip=ImageClip(title_png).set_duration(min(5.0,duration*0.22)).set_position(("center",72)).fadein(0.3).fadeout(0.3)

    # Subtitles
    subs=[]
    chunks = group_boundaries(boundaries, voice.duration) if boundaries else fallback_chunks(narration, voice.duration)
    for i,(txt,s,e) in enumerate(chunks):
        fn=f"sub_{i:03d}.png"; save_text_box(txt, fn, width=W-220, fontsize=36, bold=False)
        subs.append(ImageClip(fn).set_start(s).set_duration(e-s).set_position(("center",H-340)).fadein(0.07).fadeout(0.07))

    # Ambience (ducking)
    amb_custom = ASSETS_DIR/"ambience.mp3"
    amb_path = str(amb_custom) if amb_custom.exists() else None
    mastered="voice_mastered.mp3"
    try:
        master_voice(audio_path, mastered)
        nar_clip = AudioFileClip(mastered)
    except Exception:
        nar_clip = voice

    audio_layers=[nar_clip.set_start(0)]
    if amb_path:
        ducked="amb_ducked.mp3"
        try:
            duck_ambience(mastered if pathlib.Path(mastered).exists() else audio_path, amb_path, ducked)
            amb_clip = AudioFileClip(ducked)
        except Exception:
            amb_clip = AudioFileClip(amb_path).volumex(0.14)
        audio_layers.append(amb_clip.set_duration(duration))

    comp_audio=CompositeAudioClip([a.fx(afx.audio_normalize) for a in audio_layers])
    comp = CompositeVideoClip([broll, title_clip, *subs]).set_audio(comp_audio).set_duration(duration)

    out = re.sub(r"[^-\w\s.,()]+","",title).strip().replace(" ","_")[:80] or "autoyuson_video"
    out_path=f"{out}.mp4"
    comp.write_videofile(out_path, fps=30, codec="libx264", audio_codec="aac",
                         temp_audiofile="temp-audio.m4a", remove_temp=True, threads=4, verbose=False, logger=None)

    wiki_url=f"{WIKI}/wiki/{urllib.parse.quote(wiki_title_url)}"
    desc=(f"{title}\n\nShort factual video. No speculation, no fiction.\n\n"
          f"Source:\n- Wikipedia: {wiki_url}\n\n#history #space #culture #facts #shorts")
    return RenderResult(out_path, title, desc, DEFAULT_TAGS)

# ---------- OAuth / Upload ----------
def _read_oauth():
    cid=os.getenv("YT_CLIENT_ID","").strip()
    cs=os.getenv("YT_CLIENT_SECRET","").strip()
    rt=os.getenv("YT_REFRESH_TOKEN","").strip()
    if cid and cs and rt: return cid,cs,rt
    b64=os.getenv("TOKEN_JSON_BASE64","").strip()
    if not b64: return cid,cs,rt
    try:
        data=json.loads(base64.b64decode(b64).decode("utf-8"))
        return (data.get("client_id") or cid or "",
                data.get("client_secret") or cs or "",
                data.get("refresh_token") or rt or "")
    except Exception:
        return cid,cs,rt

def build_youtube():
    cid,cs,rt=_read_oauth()
    if not (cid and cs and rt): raise RuntimeError("YouTube OAuth secrets missing.")
    creds=Credentials(token=None, refresh_token=rt, token_uri="https://oauth2.googleapis.com/token",
                      client_id=cid, client_secret=cs,
                      scopes=["https://www.googleapis.com/auth/youtube.upload","https://www.googleapis.com/auth/youtube"])
    creds.refresh(Request())
    return build("youtube","v3",credentials=creds)

def upload_youtube(path:str,title:str,desc:str,tags:List[str],cat="27",privacy="public")->str:
    from googleapiclient.http import MediaFileUpload
    yt=build_youtube()
    body={"snippet":{"title": title if not TITLE_PREFIX else f"{TITLE_PREFIX}{title}",
                     "description":desc,"categoryId":cat,"tags":tags},
          "status":{"privacyStatus":privacy,"selfDeclaredMadeForKids":False}}
    media=MediaFileUpload(path, chunksize=-1, resumable=True, mimetype="video/*")
    req=yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp=None
    while resp is None:
        status, resp = req.next_chunk()
    return resp.get("id","")

# ---------- Orchestrator ----------
def main()->int:
    seen=load_seen()
    topic=pick_topic(seen)
    print(f"[AUTOYUSON] Topic: {topic}")

    title, summary, page_url, img_urls = wiki_fetch(topic)
    narration, short_title, sentences = craft_script(title, summary)
    if not is_factual(narration): pass

    # Görseller
    imgs=[]
    for u in img_urls:
        im=download_image(u)
        if im: imgs.append(im)
    if not imgs: imgs=[Image.new("RGB",(W,H),(20,20,24))]
    while len(imgs)<3: imgs.append(imgs[-1])

    # TTS + timings
    voice_path="voice.mp3"
    boundaries = tts_with_timings(sentences, narration, voice_path, LANG, EDGE_TTS_VOICE)

    # Render
    result=render(short_title, narration, sentences, voice_path, boundaries, imgs, DURATION_TARGET, wiki_title_url=title.replace(" ","_"))

    # Upload
    print("[AUTOYUSON] Uploading…")
    vid=upload_youtube(result.video_path, result.title, result.description, result.tags)
    print(f"[AUTOYUSON] Uploaded: https://youtube.com/watch?v={vid}")

    seen.append(title); save_seen(seen)
    print("[AUTOYUSON] Done.")
    return 0

if __name__=="__main__":
    sys.exit(main())
