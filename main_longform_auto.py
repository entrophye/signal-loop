import os, random, uuid, math, textwrap, json, time
return VideoClip(lambda t: procedural_frame(t), duration=dur).resize((1080,1920))


# -------- YouTube


def youtube_service():
client_id = os.getenv('YOUTUBE_CLIENT_ID')
client_secret = os.getenv('YOUTUBE_CLIENT_SECRET')
refresh_token = os.getenv('YOUTUBE_REFRESH_TOKEN')
token_uri = 'https://oauth2.googleapis.com/token'
creds = Credentials(None, refresh_token=refresh_token, token_uri=token_uri, client_id=client_id, client_secret=client_secret, scopes=['https://www.googleapis.com/auth/youtube.upload'])
creds.refresh(Request())
return build('youtube','v3',credentials=creds)




def upload_to_youtube(path_mp4, title, desc, tags):
youtube = youtube_service()
body = {
'snippet': {'title': title,'description': desc,'tags': tags,'categoryId': '24'},
'status': {'privacyStatus': 'public','selfDeclaredMadeForKids': False}
}
media = MediaFileUpload(path_mp4, chunksize=-1, resumable=True)
return youtube.videos().insert(part=','.join(body.keys()), body=body, media_body=media).execute().get('id')


# -------- Orchestrator


def run_pipeline():
uid = uuid.uuid4().hex[:6]


# build narrative (autonomous)
script_text, scene_caps = make_script(total_sentences=random.randint(6,9))


# TTS + sound design
raw_path = str(OUT / f"narr_{uid}.mp3")
gTTS(text=script_text, lang='en').save(raw_path)
voice_path = tts_and_design(script_text, raw_path)


# backgrounds: try fetch, else procedural
fetched = fetch_backgrounds(limit=8)
bg_files = list(OVERLAYS.glob('*.jpg')) + list(OVERLAYS.glob('*.png'))


# assemble scenes
scene_count = random.randint(SCENES_MIN, SCENES_MAX)
per_scene = VIDEO_DURATION / scene_count
clips = []
for i in range(scene_count):
if bg_files:
base = scene_from_img(random.choice(bg_files), per_scene)
else:
base = scene_procedural(per_scene)
cap = scene_caps[i % len(scene_caps)]
cap_img = caption_png(cap, width=1000)
cap_clip = ImageClip(cap_img).set_duration(per_scene).set_position(('center','center'))
clips.append(CompositeVideoClip([base, cap_clip]))


# stitch (simple concatenation)
final = clips[0]
for c in clips[1:]:
final = CompositeVideoClip([final.set_end(final.duration), c.set_start(final.duration)])
final = final.set_duration(VIDEO_DURATION)


# audio attach
aclip = AudioFileClip(voice_path)
final = final.set_audio(aclip)


# render
mp4 = str(OUT / f"signal_auto_{uid}.mp4")
final.write_videofile(mp4, fps=30, codec='libx264', audio_codec='aac')


# evolve seeds with today's captions
evolve_and_store(scene_caps)


# upload
n = datetime.utcnow().strftime('%Y-%m-%d')
title = f"SIGNAL ARCHIVE — {n} — {random.choice(['The Pulse Beneath','Whisper Archive','Eclipsera'])}"
desc = ("From the T-3012 archives. God is not gone. He's buried.


"
"Automated longform transmission.")
tags = ["godfinders","ai","cosmic horror","existential","shorts"]
vid = upload_to_youtube(mp4, title, desc, tags)
print('Uploaded video ID:', vid)


if __name__ == '__main__':
run_pipeline()
