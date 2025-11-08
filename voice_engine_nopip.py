# -*- coding: utf-8 -*-
"""
pip'siz TTS: stdlib + ffmpeg.
Öncelik: ElevenLabs REST (ELEVEN_API_KEY gerek), yoksa espeak-ng/espeak/say fallback.
Cümleleri parçalayıp her parça için TTS -> aralara kısa sessizlik -> concat -> loudnorm.

Kullanım:
    from voice_engine_nopip import synthesize_smart
    wav = synthesize_smart(text, lang="tr")  # WAV tam yolu döner
"""
import os, re, json, uuid, tempfile, subprocess, shutil, time
from urllib import request, parse, error

# --------- Basit ton analizi ----------
EMO_MAP = {
    "melancholic": {
        "tr": ["hüzün","keder","yitmek","yorgun","sessiz","karanlık","yalnız","ölüm","hiçlik"],
        "en": ["melancholy","sorrow","loss","quiet","dark","alone","void","death"],
    },
    "mysterious": {
        "tr": ["gizem","sır","esrar","bilinmez","sis","işaret"],
        "en": ["mystery","enigmatic","unknown","fog","omen","sign"],
    },
    "divine": {
        "tr": ["ilahi","tanrı","kutsal","ayin","merhamet","lütuf","kehanet"],
        "en": ["divine","god","sacred","liturgy","mercy","grace","prophecy"],
    },
    "cinematic": {
        "tr": ["epik","yükselen","destansı","görkem","sinematik","yıldız","ufuk"],
        "en": ["epic","rising","majestic","cinematic","stars","horizon"],
    },
    "neutral": {
        "tr": ["bilgi","tarih","gerçek","açıklama","tanım","kısa"],
        "en": ["info","history","fact","explain","definition","brief"],
    },
}

def analyze_tone(text):
    t = text.lower()
    scores = {k:0 for k in EMO_MAP}
    for tone,langs in EMO_MAP.items():
        for w in (langs["tr"]+langs["en"]):
            if w in t: scores[tone]+=1
    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    conf = (scores[best]/total)
    if scores[best]==0: best, conf = "neutral", 0.35
    return best, max(0.4, min(1.0, conf))

def select_preset(tone, lang="tr"):
    """
    ElevenLabs tarafında 'hazır ses' isimlerinden biri seçiliyor.
    Voice id'yi isme göre çözeceğiz (dinamik).
    """
    tone = tone.lower()
    if tone == "melancholic":
        return {"name":"Elli","stability":0.5,"similarity_boost":0.95,"style":0.35,"use_speaker_boost":True}
    if tone == "mysterious":
        return {"name":"Adam","stability":0.35,"similarity_boost":0.90,"style":0.40,"use_speaker_boost":True}
    if tone == "divine":
        return {"name":"Antoni","stability":0.45,"similarity_boost":0.92,"style":0.45,"use_speaker_boost":True}
    if tone == "cinematic":
        return {"name":"Antoni","stability":0.55,"similarity_boost":0.90,"style":0.35,"use_speaker_boost":True}
    return {"name":"Adam","stability":0.60,"similarity_boost":0.90,"style":0.25,"use_speaker_boost":True}

# --------- Yardımcılar ----------
def have_prog(name):
    return shutil.which(name) is not None

def require_ffmpeg():
    if not have_prog("ffmpeg"):
        raise RuntimeError("ffmpeg bulunamadı. Lütfen ffmpeg kurun.")

def tmpfile(suffix):
    return os.path.join(tempfile.gettempdir(), f"ve_{uuid.uuid4().hex}{suffix}")

def smart_chunk(text, max_chars=200):
    # cümle bazlı parçalama
    parts = re.split(r'([\.!\?…]+)', text.strip())
    chunks, buf = [], ""
    for p in parts:
        if len(buf)+len(p) <= max_chars:
            buf += p
        else:
            if buf.strip(): chunks.append(buf.strip())
            buf = p
    if buf.strip(): chunks.append(buf.strip())
    # küçük parçaları birleştir
    merged, cur = [], ""
    for s in chunks:
        if len(cur)+len(s) < 80:
            cur = (cur+" "+s).strip()
        else:
            if cur: merged.append(cur)
            cur = s
    if cur: merged.append(cur)
    return [m for m in merged if m]

# --------- ElevenLabs stdlib istemci ----------
ELEVEN_API = "https://api.elevenlabs.io/v1"

def eleven_headers(api_key):
    return {
        "accept":"audio/mpeg",
        "content-type":"application/json",
        "xi-api-key": api_key
    }

def eleven_voice_id_by_name(api_key, name):
    # cache
    cache = os.path.join(tempfile.gettempdir(), "ve_eleven_voices.json")
    voices = None
    if os.path.exists(cache) and (time.time() - os.path.getmtime(cache) < 3600):
        try:
            voices = json.load(open(cache,"r",encoding="utf-8"))
        except Exception:
            voices = None
    if voices is None:
        req = request.Request(ELEVEN_API + "/voices", headers={"xi-api-key": api_key})
        with request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        voices = { v["name"]: v["voice_id"] for v in data.get("voices",[]) }
        json.dump(voices, open(cache,"w",encoding="utf-8"))
    return voices.get(name)

def tts_eleven_chunk(api_key, voice_id, text, settings, model_id="eleven_monolingual_v1"):
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": settings["stability"],
            "similarity_boost": settings["similarity_boost"],
            "style": settings["style"],
            "use_speaker_boost": settings["use_speaker_boost"]
        }
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{ELEVEN_API}/text-to-speech/{voice_id}"
    req = request.Request(url, data=data, headers=eleven_headers(api_key), method="POST")
    out_mp3 = tmpfile(".mp3")
    with request.urlopen(req) as resp, open(out_mp3, "wb") as f:
        f.write(resp.read())
    return out_mp3

def convert_to_wav(in_path, out_path):
    # 44.1k mono wav
    cmd = [
        "ffmpeg","-y","-i", in_path,
        "-ar","44100","-ac","1",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def make_silence_wav(duration_sec=0.28):
    out = tmpfile(".wav")
    cmd = [
        "ffmpeg","-y",
        "-f","lavfi","-i", f"anullsrc=r=44100:cl=mono",
        "-t", str(duration_sec),
        out
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out

def concat_wavs(wavs, out_path):
    # concat demuxer
    listfile = tmpfile(".txt")
    with open(listfile,"w",encoding="utf-8") as f:
        for w in wavs:
            f.write(f"file '{w}'\n")
    cmd = ["ffmpeg","-y","-f","concat","-safe","0","-i",listfile,"-c","copy", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def loudnorm(in_path, out_path, target_i=-14.0):
    # EBU R128 normalize
    cmd = [
        "ffmpeg","-y","-i", in_path,
        "-af", f"loudnorm=I={target_i}:TP=-1.0:LRA=11:measured_I=-99:print_format=summary",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# --------- Yerel fallback: espeak-ng / espeak / say ----------
def tts_local_chunk_espeak(text, lang="tr"):
    voice = "tr" if lang.startswith("tr") else "en"
    out_wav = tmpfile(".wav")
    if have_prog("espeak-ng"):
        cmd = ["espeak-ng","-v",voice,"-s","165","-p","40","-w", out_wav, text]
    elif have_prog("espeak"):
        cmd = ["espeak","-v",voice,"-s","165","-p","40","-w", out_wav, text]
    elif have_prog("say"):  # macOS
        tmp_aiff = tmpfile(".aiff")
        cmd = ["say","-v","Yelda" if voice=="tr" else "Daniel","-o", tmp_aiff, text]
        subprocess.run(cmd, check=True)
        convert_to_wav(tmp_aiff, out_wav)
        try: os.remove(tmp_aiff)
        except: pass
        return out_wav
    else:
        raise RuntimeError("Hiçbir lokal TTS bulunamadı (espeak-ng/espeak/say).")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_wav

# --------- Üst seviye: hepsi bir arada ----------
def synthesize_smart(text, lang="tr", prefer="elevenlabs"):
    """
    Metni analiz eder → preset seçer → (ElevenLabs varsa) parça parça TTS → nefes ekle → concat → loudnorm.
    Yoksa lokal TTS'e düşer.
    Dönen: final WAV yolu.
    """
    require_ffmpeg()
    tone, _ = analyze_tone(text)
    preset = select_preset(tone, lang=lang)
    chunks = smart_chunk(text, max_chars=220)

    tmp_parts = []
    silence = make_silence_wav(0.28)

    used_cloud = False
    api_key = os.getenv("ELEVEN_API_KEY")
    voice_id = None

    if prefer == "elevenlabs" and api_key:
        try:
            voice_id = eleven_voice_id_by_name(api_key, preset["name"])
            if not voice_id:
                # ismi çözemedi, ilk bulduğu sesi kullan
                voice_id = eleven_voice_id_by_name(api_key, preset["name"]) or ""
        except Exception:
            voice_id = None

    for i, ch in enumerate(chunks):
        if voice_id:  # ElevenLabs yolu
            try:
                mp3 = tts_eleven_chunk(api_key, voice_id, ch, preset,
                                       model_id="eleven_multilingual_v2")
                wav = tmpfile(".wav")
                convert_to_wav(mp3, wav)
                tmp_parts.append(wav)
                used_cloud = True
            except Exception:
                # cloud hata -> lokal dene
                wav = tts_local_chunk_espeak(ch, lang=lang)
                tmp_parts.append(wav)
        else:
            # direkt lokal
            wav = tts_local_chunk_espeak(ch, lang=lang)
            tmp_parts.append(wav)

        # son parça değilse nefes
        if i < len(chunks)-1:
            tmp_parts.append(silence)

    merged = tmpfile(".wav")
    concat_wavs(tmp_parts, merged)

    final_wav = tmpfile(".wav")
    loudnorm(merged, final_wav, target_i=-14.0)

    # temizlik (sessizlik dosyasını silme; başka çağrılarda da kullanılabilir)
    try:
        for p in tmp_parts:
            if os.path.exists(p) and p != silence:
                os.remove(p)
        if os.path.exists(merged):
            os.remove(merged)
    except Exception:
        pass

    return final_wav
