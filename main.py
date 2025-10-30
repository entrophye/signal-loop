import os, random, time, textwrap, uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, TextClip
from moviepy.video.fx.all import crop
from pydub import AudioSegment
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

load_dotenv()
ROOT = Path(__file__).parent
DATA = ROOT / 'data'
OVERLAYS = DATA / 'overlays'
SEEDS = DATA / 'seeds_godfinders.txt'
OUT = ROOT / 'out'
OUT.mkdir(exist_ok=True)

LANG = os.getenv('LANG', 'en')
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'gtts')

# ===== Content Generation =====
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
    seeds = [s.strip() for s in SEEDS.read_text(encoding='utf-8').splitlines() if s.strip()]
    return seeds if seeds else [
        "God is not gone. He's buried.",
        "The Pulse counts backwards.",
        "Our bones learned a new alphabet.",
    ]


SEED_LINES = load_seeds()


def make_monologue(max_len=85):
    hook = random.choice(HOOKS)
    line1, line2 = random.sample(SEED_LINES, 2)
    turn = random.choice(TURNS)
    txt = random.choice(G_TEMPLATES).format(hook=hook, line1=line1, line2=line2, turn=turn)
    txt = ' '.join(txt.split())
    if len(txt) > max_len:
        txt = txt[:max_len - 1] + '…'
    return txt


# ===== TTS =====
def synth_voice(text, out_mp3):
    if TTS_PROVIDER == 'gtts':
        gTTS(text=text, lang='en').save(out_mp3)
        return out_mp3
    else:
        # Placeholder for ElevenLabs etc.
        gTTS(text=text, lang='en').save(out_mp3)
        return out_mp3


# ===== Video =====
def render_video(text, voice_mp3, out_mp4, duration=10):
    bg_list = list(OVERLAYS.glob('*.jpg'))
    bg_path = random.choice(bg_list) if bg_list else None

    if not bg_path:
        # fallback: solid color pulse
        import numpy as np
        from moviepy.editor import VideoClip

        def make_frame(t):
            base = int(8 + 4 * abs((t % 2) - 1))
            val = 10 + base
            return np.uint8([[[val, val, val]]])

        clip = VideoClip(make_frame, duration=duration).resize((1080, 1920))
    else:
        clip = ImageClip(str(bg_path)).resize(height=1920).set_duration(duration)
        # gentle zoom
        w, h = clip.size
        zoom = clip.fx(crop, x1=0, y1=0, x2=w, y2=h).resize(lambda t: 1.02 + 0.005 * t)
        clip = zoom

    audio = AudioFileClip(voice_mp3)
    clip = clip.set_audio(audio)

    # subtitles
    txt = TextClip(
        textwrap.fill(text, 28),
        fontsize=64,
        color='white',
        font='DejaVu-Sans',
        method='caption',
        size=(1000, None),
        align='center'
    )
    txt = txt.set_position(('center', 'center')).set_duration(audio.duration)

    result = CompositeVideoClip([clip, txt])
    result.write_videofile(out_mp4, fps=30, codec='libx264', audio_codec='aac')


# ===== YouTube Upload =====
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


def upload_to_youtube(path_mp4, title, desc, tags):
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


# ===== Orchestration =====
def run_pipeline():
    uid = uuid.uuid4().hex[:6]
    text = make_monologue()
    mp3 = str(OUT / f'signal_{uid}.mp3')
    mp4 = str(OUT / f'signal_{uid}.mp4')

    synth_voice(text, mp3)

    # pad to ~10s if too short
    a = AudioSegment.from_file(mp3)
    if len(a) < 8000:
        silence = AudioSegment.silent(duration=8000 - len(a))
        a = a + silence
        a.export(mp3, format='mp3')

    render_video(text, mp3, mp4, duration=10)

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
    # If running under GitHub Actions, just run once.
    run_pipeline()
