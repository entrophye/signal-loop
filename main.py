import os
import random
import textwrap
import uuid
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment

# MoviePy
import numpy as np
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, VideoClip
from moviepy.video.fx.all import crop

# PIL (ImageMagick bağımlılığı yok)
from PIL import Image, ImageDraw, ImageFont, ImageOps

# YouTube API
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# ============== ENV / PATHS ==============
load_dotenv()
ROOT = Path(__file__).parent
DATA = ROOT / 'data'
OVERLAYS = DATA / 'overlays'
SEEDS = DATA / 'seeds_godfinders.txt'
OUT = ROOT / 'out'
OUT.mkdir(exist_ok=True)

LANG = os.getenv('LANG', 'en')
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'gtts')

# ============== CONTENT GEN ==============
G_TEMPLATES = [
    "{hook} {line1} {turn} {line2}",
    "{hook} {line1} {line2}",
    "{hook} {turn} {line1}",
]
HOOKS = [
    "Transmission begins:",
    "Archive fragment:",
    "Field log:",
    "After the collapse:",
]
TURNS = [
    "We kept digging.",
    "The signal grew teeth.",
    "Silence replied in numbers.",
    "No maps survived the light.",
]


def load_seeds():
    if SEEDS.exists():
        lines = [s.strip() for s in SEEDS.read_text(encoding='utf-8').splitlines() if s.strip()]
        if lines:
            return lines
    return [
        "God is not gone. He's buried.",
        "The Pulse counts backwards.",
        "Our bones learned a new alphabet.",
    ]


SEED_LINES = load_seeds()


def make_monologue(max_len: int = 85) -> str:
    hook = random.choice(HOOKS)
    line1, line2 = random.sample(SEED_LINES, 2)
    turn = random.choice(TURNS)
    txt = random.choice(G_TEMPLATES).format(hook=hook, line1=line1, line2=line2, turn=turn)
    txt = ' '.join(txt.split())
    if len(txt) > max_len:
        txt = txt[:max_len - 1] + '…'
    return txt


# ============== TTS ==============
def synth_voice(text: str, out_mp3: str) -> str:
    # Varsayılan: gTTS (ücretsiz). İleride ElevenLabs eklenebilir.
    gTTS(text=text, lang='en').save(out_mp3)
    return out_mp3


# ============== CAPTION (PIL) ==============
def make_caption_image(
    text: str,
    width: int = 1000,
    padding: int = 30,
    bg=(0, 0, 0, 0),
    fg=(255, 255, 255, 255),
) -> str:
    wrapped = textwrap.fill(text, 28)
    # Font: DejaVuSans varsa onu kullan, yoksa default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 64)
    except Exception:
        font = ImageFont.load_default()

    # Metin ölçümü
    dummy = Image.new("RGBA", (width, 10), bg)
    d = ImageDraw.Draw(dummy)
    bbox = d.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=6)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    img = Image.new("RGBA", (width + 2 * padding, h + 2 * padding), bg)

    # Gölge katmanı (okunurluk)
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    for ox, oy in ((1, 1), (2, 2), (-1, 1), (1, -1), (-2, -2)):
        sd.multiline_text(
            (img.width // 2 + ox, padding + oy),
            wrapped,
            font=font,
            fill=(0, 0, 0, 140),
            align="center",
            anchor="ma",
            spacing=6,
        )

    # Metni asıl katmana çiz
    d2 = ImageDraw.Draw(img)
    d2.multiline_text(
        (img.width // 2, padding),
        wrapped,
        font=font,
        fill=fg,
        align="center",
        anchor="ma",
        spacing=6,
    )

    # Gölge + metin birleştir
    composed = Image.alpha_composite(shadow, img)

    out_path = str(OUT / f"caption_{uuid.uuid4().hex[:6]}.png")
    composed.save(out_path)
    return out_path


# ============== VIDEO RENDER ==============
def render_video(text: str, voice_mp3: str, out_mp4: str, duration: int = 10) -> None:
    # Arkaplan: varsa jpg seç, yoksa sade puls
    bg_list = list(OVERLAYS.glob('*.jpg'))
    bg_path = random.choice(bg_list) if bg_list else None

    if not bg_path:
        # Basit gri puls
        def make_frame(t):
            base = int(8 + 4 * abs((t % 2) - 1))
            val = 10 + base
            # 1x1 görüntü → resize ile 1080x1920
            return np.uint8([[[val, val, val]]])

        clip = VideoClip(make_frame, duration=duration).resize((1080, 1920))
    else:
        clip = ImageClip(str(bg_path)).resize(height=1920).set_duration(duration)
        w, h = clip.size
        zoom = clip.fx(crop, x1=0, y1=0, x2=w, y2=h).resize(lambda t: 1.02 + 0.005 * t)
        clip = zoom

    audio = AudioFileClip(voice_mp3)
    clip = clip.set_audio(audio)

    # Altyazı PNG (PIL) → overlay
    cap_path = make_caption_image(text, width=1000)
    caption = ImageClip(cap_path).set_duration(audio.duration).set_position(('center', 'center'))

    result = CompositeVideoClip([clip, caption])
    result.write_videofile(out_mp4, fps=30, codec='libx264', audio_codec='aac')


# ============== YOUTUBE UPLOAD ==============
def youtube_service():
    client_id = os.getenv('YOUTUBE_CLIENT_ID')
    client_secret = os.getenv('YOUTUBE_CLIENT_SECRET')
    refresh_token = os.getenv('YOUTUBE_REFRESH_TOKEN')
    token_uri = 'https://oauth2.googleapis.com/token'

    creds = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri=token_uri,
        client_id=client_id,
        client_secret=client_secret,
        scopes=['https://www.googleapis.com/auth/youtube.upload']
    )
    creds.refresh(Request())
    return build('youtube', 'v3', credentials=creds)


def upload_to_youtube(path_mp4: str, title: str, desc: str, tags: list) -> str:
    youtube = youtube_service()
    body = {
        'snippet': {
            'title': title,
            'description': desc,
            'tags': tags,
            'categoryId': '24'  # Entertainment
        },
        'status': {
            'privacyStatus': 'public',
            'selfDeclaredMadeForKids': False
        }
    }
    media = MediaFileUpload(path_mp4, chunksize=-1, resumable=True)
    request = youtube.videos().insert(part=','.join(body.keys()), body=body, media_body=media)
    response = request.execute()
    return response.get('id')


# ============== ORCHESTRATION ==============
def run_pipeline():
    uid = uuid.uuid4().hex[:6]
    text = make_monologue()

    mp3 = str(OUT / f'signal_{uid}.mp3')
    mp4 = str(OUT / f'signal_{uid}.mp4')

    # TTS
    synth_voice(text, mp3)

    # Ses 8s'ten kısaysa 8s'e doldur
    a = AudioSegment.from_file(mp3)
    if len(a) < 8000:
        a = a + AudioSegment.silent(duration=8000 - len(a))
        a.export(mp3, format='mp3')

    # Video
    render_video(text, mp3, mp4, duration=10)

    # Başlık/açıklama
    n = datetime.utcnow().strftime('%Y-%m-%d')
    title = f"SIGNAL — {n} — {random.choice(['The Pulse Beneath','Whisper Archive','Eclipsera'])}"
    desc = (
        "From the T-3012 archives. God is not gone. He's buried.\n\n"
        "Godfinders project — automated transmission loop."
    )
    tags = ["godfinders", "ai", "shorts", "cosmic horror", "existential"]
    if os.getenv('CHANNEL_KEYWORDS'):
        tags += [t.strip() for t in os.getenv('CHANNEL_KEYWORDS').split(',') if t.strip()]

    vid = upload_to_youtube(mp4, title, desc, tags)
    print('Uploaded video ID:', vid)


if __name__ == '__main__':
    run_pipeline()
