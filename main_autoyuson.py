# -*- coding: utf-8 -*-
"""
AUTOYUSON V5 — Cinematic Factual Shorts (History / Space / Culture)

Kalite odaklı tam yeniden yazım:
- TTS: ElevenLabs REST (SDK yok), yoksa gTTS. FORCE_TTS=edge|gtts|eleven (edge devre dışı: bu dosya edge kullanmıyor)
- Tempo: atempo 0.92 + cümleler arası 320ms sessizlik (ffmpeg ile)
- Altyazı: konuşma süresine oranlı, min 1.2s / max 4.0s, okunur hız
- Görseller: Wikimedia 'original' alanı, min 1600x1600, akıllı ölçek + hafif sharpen + sıcak grade + vignette + grain
- Müzik: assets/music/*.mp3 rastgele, düşük volüm, fade-in/out
- Encode: crf 20, preset medium, yuv420p
- Loglar: TTS seçimi, elenen görseller, altyazı metrikleri

ENV:
  YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN  (veya TOKEN_JSON_BASE64)
  LANGUAGE=en
  VIDEO_DURATION=55
  TITLE_PREFIX=
  TOPICS_FILE=topics.txt
  ELEVEN_API_KEY=<key>
  ELEVEN_VOICE=Elli|Adam|Antoni|... (isteğe bağlı, ad ile eşleşme yapılır)
  PRIVACY_STATUS=public|unlisted|private
  DISABLE_UPLOAD=true|false
  FORCE_TTS=eleven|gtts
"""

import os, io, re, json, base64, random, pathlib, sys, urllib.parse, math, time, glob, subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

try:
    from gtts import gTTS
    _HAS_GTTS=True
except Exception:
    _HAS_GTTS=False

try:
    from moviepy.editor import (ImageClip, AudioFileClip, CompositeVideoClip,
                                CompositeAudioClip, afx)
    _HAS_MOVIEPY=True
except Exception:
    _HAS_MOVIEPY=False

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    _HAS_GOOGLE=True
except Exception:
    _HAS_GOOGLE=False

# -------- Config
LANG = (os.getenv("LANGUAGE","en").strip() or "en")[:5]
DURATION_TARGET = int(os.getenv("VIDEO_DURATION","55"))
TITLE_PREFIX = os.getenv("TITLE_PREFIX","").strip()
PRIVACY_STATUS = (os.getenv("PRIVACY_STATUS","public").strip().lower() or "public")
DISABLE_UPLOAD = os.getenv("DISABLE_UPLOAD","").strip().lower() in ("1","true","yes")
FORCE_TTS = os.getenv("FORCE_TTS","").strip().lower()  # 'eleven' | 'gtts' | ''
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY","").strip()
ELEVEN_VOICE = os.getenv("ELEVEN_VOICE","").strip() or "Elli"

DATA = pathlib.Path("data"); DATA.mkdir(exist_ok=True, parents=True)
SEEN = DATA/"seen_topics.json"
FONTS = pathlib.Path("fonts"); FONTS.mkdir(exist_ok=True)
ASSETS = pathlib.Path("assets"); ASSETS.mkdir(exist_ok=True)

W,H = 1080,1920
WIKI="https://en.wikipedia.org"; REST=f"{WIKI}/api/rest_v1"; API=f"{WIKI}/w/api.php"
UA={"User-Agent":"autoyuson-v5 (+github)"}

DEFAULT_TAGS=["history","space","astronomy","archaeology","culture","documentary","facts","shorts"]
CURATED_TOPICS=[
    "Rosetta Stone","Dead Sea Scrolls","Hagia Sophia","Voyager Golden Record","Terracotta Army","Göbekli Tepe",
    "Library of Alexandria","Code of Hammurabi","Magna Carta","Edict of Milan","Battle of Thermopylae",
    "Byzantine Iconoclasm","Hubble's law","Pioneer 10","James Webb Space Telescope","Apollo 8 Earthrise",
    "Arecibo message","Cassini–Huygens","Venera 13","Voyager 1","Chandrasekhar limit","Nazca Lines",
    "Antikythera mechanism","I Ching","Shahnameh","Silk Road","Machu Picchu","Moai","Çatalhöyük"
]

# -------- Utils
def log(x): print(x, flush=True)

def load_seen():
    if SEEN.exists():
        try: return json.loads(SEEN.read_text(encoding="utf-8"))
        except: return []
    return []

def save_seen(lst):
    try: SEEN.write_text(json.dumps(lst[-1000:], ensure_ascii=False, indent=2), encoding="utf-8")
    except: pass

def extra_topics():
    p=pathlib.Path(os.getenv("TOPICS_FILE","topics.txt"))
    if p.exists():
        try:
            return [t.strip() for t in p.read_text(encoding="utf-8").splitlines()
                    if t.strip() and not t.strip().startswith("#")]
        except: return []
    return []

def pick_topic(seen):
    pool = extra_topics() + CURATED_TOPICS
    random.shuffle(pool)
    recent=set(seen[-300:])
    for t in pool:
        if t not in recent:
            return t
    return (pool[0] if pool else "Göbekli Tepe")

# -------- Wikipedia
def _rest_summary(title):
    u=f"{REST}/page/summary/{urllib.parse.quote(title)}"
    try:
        r=requests.get(u,headers=UA,timeout=20); 
        return r.json() if r.status_code==200 else None
    except: return None

def _rest_media(title):
    u=f"{REST}/page/media/{urllib.parse.quote(title)}"
    try:
        r=requests.get(u,headers=UA,timeout=20); 
        return r.json() if r.status_code==200 else None
    except: return None

def _opensearch(q):
    try:
        r=requests.get(API,headers=UA,timeout=20,params={
            "action":"opensearch","search":q,"limit":5,"namespace":0,"format":"json"
        })
        if r.status_code!=200: return []
        d=r.json(); return d[1] if isinstance(d,list) and len(d)>=2 else []
    except: return []

def wiki_fetch(topic):
    cands=[topic]+_opensearch(topic)
    for cand in cands:
        s=_rest_summary(cand)
        if not s: continue
        title=s.get("title") or cand
        extract=s.get("extract") or ""
        page_url=s.get("content_urls",{}).get("desktop",{}).get("page") or f"{WIKI}/wiki/{title.replace(' ','_')}"
        imgs=[]
        m=_rest_media(title)
        if m and "items" in m:
            for it in m["items"]:
                if it.get("type")!="image": continue
                # 'original' varsa onu kullan
                orig=it.get("original",{})
                src=None
                if orig and orig.get("source"):
                    src=orig["source"]
                    w=orig.get("width",0); h=orig.get("height",0)
                    if min(w,h)<1600:  # düşük çözünürlük ele
                        src=None
                if not src:
                    # srcset içinde en yüksek ölçek
                    ss=sorted(it.get("srcset",[]), key=lambda x: x.get("scale",1.0))
                    if ss: src=ss[-1].get("src")
                if not src: src=it.get("src")
                if not src: continue
                low=src.lower()
                if any(b in low for b in ["logo","icon","flag","seal","map","diagram","coat_of_arms","favicon","sprite"]):
                    continue
                if not low.endswith((".jpg",".jpeg",".png",".webp")): continue
                imgs.append(src)
        thumb=s.get("thumbnail",{}).get("source")
        if thumb: imgs=[thumb]+imgs
        if extract:
            return title, extract, page_url, imgs[:10]
    raise RuntimeError("Wiki fetch failed")

def download_image(url):
    try:
        r=requests.get(url,headers=UA,timeout=30); r.raise_for_status()
        im=Image.open(io.BytesIO(r.content)).convert("RGB")
        return im
    except: return None

# -------- Script
def craft_script(title, summary):
    clean=re.sub(r"\s+"," ", summary).strip()
    cold=f"{title}. Verified, preserved, undeniable."
    hook="What changed our understanding? Evidence — not legend."
    words=clean.split()
    fact1=" ".join(words[:40]) if words else clean
    fact2=" ".join(words[40:80]) or clean
    fact3=" ".join(words[80:120]) or clean
    close="History survives by proof — and by those who keep looking."
    sents=[cold,hook,fact1,fact2,fact3,close]
    narration=" ".join(sents)
    return narration, title, sents

def factual_ok(text):
    if any(k in text.lower() for k in ["reportedly","allegedly","legend","rumor"]): return False
    n=len(text.split()); return 80<=n<=260

# -------- Fonts/Text
REG=FONTS/"NotoSans-Regular.ttf"; BLD=FONTS/"NotoSans-Bold.ttf"
def ensure_fonts():
    urls={
        REG:"https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
        BLD:"https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
    }
    for p,u in urls.items():
        if not p.exists():
            try:
                r=requests.get(u,headers=UA,timeout=30); r.raise_for_status()
                p.write_bytes(r.content)
            except: pass

def get_font(size=48,bold=False):
    try:
        return ImageFont.truetype(str(BLD if bold else REG), size=size)
    except: return ImageFont.load_default()

def text_box(text,width,fontsize,bold=False,fg=(255,255,255),bg=(0,0,0,150)):
    f=get_font(fontsize,bold)
    tmp=Image.new("RGBA",(width,1200),(0,0,0,0)); d=ImageDraw.Draw(tmp)
    words=text.split(); lines=[]; cur=""
    for w in words:
        t=(cur+" "+w).strip()
        wpx=d.textlength(t,font=f)
        if wpx>width-40:
            if cur: lines.append(cur); cur=w
            else: lines.append(w); cur=""
        else: cur=t
    if cur: lines.append(cur)
    ascent,descent=f.getmetrics() if hasattr(f,"getmetrics") else (fontsize,int(fontsize*0.25))
    lh=ascent+descent+6
    line_w=[d.textlength(x,font=f) for x in lines] or [width//2]
    box_w=min(width, int(max(line_w))+36); box_h=lh*len(lines)+28
    img=Image.new("RGBA",(width,box_h),(0,0,0,0))
    rect=Image.new("RGBA",(box_w,box_h),bg)
    img.alpha_composite(rect,((width-box_w)//2,0))
    y=14; draw=ImageDraw.Draw(img)
    for ln in lines:
        wpx=int(draw.textlength(ln,font=f)); x=(width-wpx)//2
        draw.text((x,y),ln,font=f,fill=fg); y+=lh
    return img

def save_text(text,out,width,fontsize,bold=False):
    img=text_box(text,width,fontsize,bold); img.save(out,"PNG")

# -------- TTS (ElevenLabs REST + gTTS fallback)
ELEVEN_API="https://api.elevenlabs.io/v1"

def _eleven_headers():
    return {
        "accept":"audio/mpeg",
        "content-type":"application/json",
        "xi-api-key": ELEVEN_API_KEY
    }

def eleven_voice_id(name):
    cache=DATA/"eleven_voices.json"
    voices=None
    if cache.exists() and time.time()-cache.stat().st_mtime<3600:
        try: voices=json.loads(cache.read_text(encoding="utf-8"))
        except: voices=None
    if voices is None:
        r=requests.get(f"{ELEVEN_API}/voices",headers={"xi-api-key":ELEVEN_API_KEY},timeout=20)
        r.raise_for_status()
        data=r.json(); voices={v["name"]:v["voice_id"] for v in data.get("voices",[])}
        cache.write_text(json.dumps(voices,ensure_ascii=False,indent=2),encoding="utf-8")
    return voices.get(name)

def tts_eleven(sentences, out_mp3, voice_name):
    vid=eleven_voice_id(voice_name)
    if not vid: raise RuntimeError("ElevenLabs voice id bulunamadı.")
    # cümle bazlı üret ve aralara 320ms sessizlik ekle
    parts=[]
    for i, s in enumerate(sentences):
        payload={
            "text": s,
            "model_id": "eleven_multilingual_v2",
            "voice_settings":{
                "stability":0.45, "similarity_boost":0.92, "style":0.45, "use_speaker_boost":True
            }
        }
        rq=requests.post(f"{ELEVEN_API}/text-to-speech/{vid}",
                         headers=_eleven_headers(), data=json.dumps(payload).encode("utf-8"), timeout=60)
        rq.raise_for_status()
        p=f"piece_{i}.mp3"; open(p,"wb").write(rq.content); parts.append(p)
        # son parça değilse sessizlik
        if i<len(sentences)-1:
            sil=f"sil_{i}.wav"
            subprocess.run(["ffmpeg","-y","-f","lavfi","-i","anullsrc=r=44100:cl=mono","-t","0.32",sil],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            parts.append(sil)
    # concat
    lst="concat.txt"; open(lst,"w",encoding="utf-8").write("\n".join([f"file '{p}'" for p in parts]))
    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-c","copy",out_mp3],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    # tempo (yavaşlat)
    slowed="tmp_slow.mp3"
    subprocess.run(["ffmpeg","-y","-i",out_mp3,"-filter:a","atempo=0.92",slowed],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    os.replace(slowed,out_mp3)
    # temizlik
    for p in parts:
        try: os.remove(p)
        except: pass
    try: os.remove(lst)
    except: pass

def tts_gtts(text, out_mp3, lang):
    if not _HAS_GTTS: raise RuntimeError("gTTS yok.")
    gTTS(text=text,lang=lang,tld="com").save(out_mp3)
    # tempo yavaşlat
    slowed="tmp_slow.mp3"
    subprocess.run(["ffmpeg","-y","-i",out_mp3,"-filter:a","atempo=0.92",slowed],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    os.replace(slowed,out_mp3)

def synth_tts(sentences, narration, out_mp3, lang):
    log(f"[TTS] FORCE_TTS={FORCE_TTS or '-'} ELEVEN={'yes' if ELEVEN_API_KEY else 'no'} GTTS={'yes' if _HAS_GTTS else 'no'}")
    if FORCE_TTS=="eleven":
        if not ELEVEN_API_KEY: raise RuntimeError("FORCE_TTS=eleven fakat ELEVEN_API_KEY yok.")
        tts_eleven(sentences, out_mp3, ELEVEN_VOICE); log("[TTS] Provider=ElevenLabs"); return
    if FORCE_TTS=="gtts":
        tts_gtts(narration, out_mp3, lang); log("[TTS] Provider=gTTS"); return
    # otomatik
    if ELEVEN_API_KEY:
        try: tts_eleven(sentences, out_mp3, ELEVEN_VOICE); log("[TTS] Provider=ElevenLabs (auto)"); return
        except Exception as e:
            log(f"[TTS] ElevenLabs failed -> gTTS: {e}")
    tts_gtts(narration, out_mp3, lang); log("[TTS] Provider=gTTS (fallback)")

# -------- Audio helpers
def pick_music():
    files=glob.glob(str(ASSETS/"music" / "*.mp3"))
    return random.choice(files) if files else None

# -------- Visuals
def fit_center_crop(img, w=W, h=H):
    r=img.width/img.height; rf=w/h
    if r>rf:
        new_h=h; new_w=int(r*new_h)
    else:
        new_w=w; new_h=int(new_w/r)
    img=img.resize((new_w,new_h),Image.LANCZOS)
    L=(img.width-w)//2; T=(img.height-h)//2
    return img.crop((L,T,L+w,T+h))

def apply_grade(img):
    w,h=img.size
    graded=ImageOps.colorize(ImageOps.grayscale(img), black="#0c0c0c", white="#f1e6d7").convert("RGB")
    graded=Image.blend(img, graded, 0.20)
    # hafif sharpen
    graded=graded.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=4))
    # vignette
    mask=Image.new("L",(w,h),160); draw=ImageDraw.Draw(mask)
    m=int(min(w,h)*0.28); draw.ellipse((m,m,w-m,h-m), fill=0); mask=mask.filter(ImageFilter.GaussianBlur(85))
    overlay=Image.new("RGBA",(w,h),(0,0,0,255)); overlay.putalpha(mask)
    base=graded.convert("RGBA")
    vig=Image.alpha_composite(base, overlay).convert("RGB")
    # grain
    noise=(np.random.randn(h,w)*5).clip(-10,10).astype(np.int16)
    arr=np.array(vig).astype(np.int16); arr=np.clip(arr + noise[:,:,None],0,255).astype(np.uint8)
    return Image.fromarray(arr)

def prepare_images(img_urls):
    imgs=[]; dropped=0
    for u in img_urls:
        im=download_image(u)
        if not im: continue
        if min(im.size)<1600:
            dropped+=1; continue
        im=fit_center_crop(im, W, H)
        im=apply_grade(im)
        imgs.append(im)
        if len(imgs)>=6: break
    if not imgs:
        imgs=[Image.new("RGB",(W,H),(22,22,26))]
    log(f"[IMG] accepted={len(imgs)} dropped(lowres)={dropped}")
    return imgs

def ken_burns_frame(img, duration, z0=1.05, z1=1.12):
    from moviepy.editor import ImageClip
    tmp=f"kb_{int(time.time()*1000)}.jpg"; img.save(tmp,quality=92)
    clip=ImageClip(tmp).set_duration(duration)
    return clip.resize(lambda t: z0+(z1-z0)*(t/duration)).set_position(("center","center"))

def build_broll(images, duration):
    from moviepy.editor import CompositeVideoClip
    if len(images)==1:
        return ken_burns_frame(images[0], duration)
    per=max(3.8, min(7.0, duration/len(images))); fc=0.45
    layers=[]; t=0.0
    for im in images:
        c=ken_burns_frame(im, per).set_start(t).crossfadein(fc).crossfadeout(fc)
        layers.append(c); t+= per - fc
    return CompositeVideoClip(layers).set_duration(duration)

# -------- Subtitles (duration-proportional)
def alloc_subs(sentences, voice_duration):
    weights=[max(20,len(s)) for s in sentences]  # karakter ağırlığı
    total=sum(weights)
    # toplam video sesi içinde %92'lik kısmı altyazıya ayır, baş/son nefes için 8%
    budget=voice_duration*0.92
    times=[]
    t=voice_duration*0.04
    for i,w in enumerate(weights):
        chunk=max(1.2, min(4.0, budget*(w/total)))  # 1.2s–4.0s
        times.append((sentences[i], t, t+chunk))
        t+=chunk
    # taşmalar düzelt
    times=[(txt, max(0.3,s), min(voice_duration-0.2,e)) for (txt,s,e) in times]
    return times

# -------- Title/Desc
def sanitize_title(t): 
    t=re.sub(r"[^-\w\s.,()]+","",t).strip()
    return (t or "Untitled")[:64]

def youtube_desc(title, wiki_title):
    wiki_url=f"{WIKI}/wiki/{urllib.parse.quote(wiki_title)}"
    return (f"{title}\n\nShort factual video. No speculation, no fiction.\n\n"
            f"Source:\n- Wikipedia: {wiki_url}\n\n#history #space #culture #facts #shorts")

# -------- YouTube
def _read_oauth():
    cid=os.getenv("YT_CLIENT_ID","").strip()
    cs=os.getenv("YT_CLIENT_SECRET","").strip()
    rt=os.getenv("YT_REFRESH_TOKEN","").strip()
    if cid and cs and rt: return cid,cs,rt
    b64=os.getenv("TOKEN_JSON_BASE64","").strip()
    if b64:
        try:
            d=json.loads(base64.b64decode(b64).decode("utf-8"))
            return (d.get("client_id") or cid or "",
                    d.get("client_secret") or cs or "",
                    d.get("refresh_token") or rt or "")
        except: pass
    return cid,cs,rt

def build_youtube():
    if not _HAS_GOOGLE: raise RuntimeError("Google API client yok.")
    cid,cs,rt=_read_oauth()
    if not (cid and cs and rt): raise RuntimeError("YouTube OAuth secrets missing.")
    creds=Credentials(token=None, refresh_token=rt, token_uri="https://oauth2.googleapis.com/token",
                      client_id=cid, client_secret=cs,
                      scopes=["https://www.googleapis.com/auth/youtube.upload","https://www.googleapis.com/auth/youtube"])
    creds.refresh(Request())
    return build("youtube","v3",credentials=creds)

def upload_youtube(path,title,desc,tags,cat="27",privacy="public"):
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

# -------- Render
@dataclass
class Result: path:str; title:str; desc:str; tags:List[str]

def render_video(title, sentences, narration_mp3, images, duration_hint, wiki_title):
    from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip

    ensure_fonts()
    safe_title=sanitize_title(title)

    voice=AudioFileClip(narration_mp3)
    duration=max(min(max(duration_hint,48),75), voice.duration+0.6)

    # b-roll
    broll=build_broll(images, duration)

    # title
    title_png=f"title_{int(time.time())}.png"
    save_text(safe_title, title_png, width=W-140, fontsize=58, bold=True)
    title_clip=ImageClip(title_png).set_duration(min(5.0,duration*0.22)).set_position(("center",72)).fadein(0.3).fadeout(0.3)

    # subtitles
    subs=[]
    subs_times=alloc_subs(sentences, voice.duration)
    sub_pngs=[]
    for i,(txt,s,e) in enumerate(subs_times):
        fn=f"sub_{i:03d}.png"; sub_pngs.append(fn)
        save_text(txt, fn, width=W-220, fontsize=36, bold=False)
        subs.append(ImageClip(fn).set_start(s).set_duration(e-s).set_position(("center",H-340)).fadein(0.08).fadeout(0.08))

    # music (optional)
    music_path=pick_music()
    audio_layers=[voice.set_start(0)]
    if music_path:
        music=AudioFileClip(music_path).volumex(0.12).audio_fadein(0.6).audio_fadeout(0.8)
        audio_layers.append(music.set_duration(duration))

    comp_audio=CompositeAudioClip([a.fx(afx.audio_normalize) for a in audio_layers])
    comp=(CompositeVideoClip([broll, title_clip, *subs]).set_audio(comp_audio).set_duration(duration))

    out=f"{safe_title.replace(' ','_')[:80]}.mp4"
    comp.write_videofile(
        out, fps=30, codec="libx264", audio_codec="aac",
        temp_audiofile="temp-audio.m4a", remove_temp=True, threads=4, verbose=False, logger=None,
        ffmpeg_params=["-crf","20","-preset","medium","-pix_fmt","yuv420p"]
    )

    # cleanup
    try:
        os.remove(title_png)
        for f in sub_pngs:
            if os.path.exists(f): os.remove(f)
    except: pass

    desc=youtube_desc(title, wiki_title)
    return Result(out, title, desc, DEFAULT_TAGS)

# -------- Main
def main():
    if not _HAS_MOVIEPY:
        raise RuntimeError("MoviePy gerekli.")

    seen=load_seen()
    topic=pick_topic(seen)
    log(f"[AUTOYUSON] Topic: {topic}")

    title, summary, page_url, img_urls = wiki_fetch(topic)
    narration, short_title, sentences = craft_script(title, summary)
    if not factual_ok(narration):
        log("[WARN] narration not in factual bounds; continuing.")

    # images
    images=prepare_images(img_urls)

    # TTS
    voice_mp3="voice.mp3"
    synth_tts(sentences, narration, voice_mp3, LANG)

    # render
    res=render_video(short_title, sentences, voice_mp3, images, DURATION_TARGET, wiki_title=title.replace(" ","_"))
    log(f"[AUTOYUSON] Rendered: {res.path}")

    # upload
    if DISABLE_UPLOAD:
        log("[AUTOYUSON] Upload disabled.")
    else:
        try:
            log("[AUTOYUSON] Uploading…")
            vid=upload_youtube(res.path, res.title, res.desc, res.tags, cat="27", privacy=PRIVACY_STATUS)
            log(f"[AUTOYUSON] Uploaded: https://youtube.com/watch?v={vid}")
        except Exception as e:
            log(f"[AUTOYUSON] Upload failed: {e}")

    seen.append(title); save_seen(seen)
    log("[AUTOYUSON] Done.")
    return 0

if __name__=="__main__":
    sys.exit(main())
