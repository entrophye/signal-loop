import os, random, uuid, math, textwrap
'privacyStatus': 'public',
'selfDeclaredMadeForKids': False
}
}
media = MediaFileUpload(path_mp4, chunksize=-1, resumable=True)
req = youtube.videos().insert(part=','.join(body.keys()), body=body, media_body=media)
resp = req.execute()
return resp.get('id')


# ===== Orchestrator


def run_pipeline():
uid = uuid.uuid4().hex[:6]


# Narrative
script_text, scene_captions = make_script(total_sentences=random.randint(6,9))


# TTS + sound design
raw_mp3 = str(OUT / f"narr_{uid}.mp3")
synth_voice(script_text, raw_mp3)
voice_mp3 = design_audio(raw_mp3, MIN_AUDIO_MS)


# scenes
scene_count = random.randint(SCENES_MIN, SCENES_MAX)
per_scene = VIDEO_DURATION / scene_count


bg_files = list(OVERLAYS.glob('*.jpg')) + list(OVERLAYS.glob('*.png'))
clips = []
for i in range(scene_count):
if bg_files:
img_path = random.choice(bg_files)
base = build_scene_from_image(img_path, per_scene)
else:
base = build_scene_procedural(per_scene)


# caption per scene
cap = scene_captions[i % len(scene_captions)]
cap_path = make_caption_image(cap, width=1000)
cap_clip = ImageClip(cap_path).set_duration(per_scene).set_position(('center','center'))
clips.append(CompositeVideoClip([base, cap_clip]))


# crossfades
xf = 0.6
timeline = clips[0].fadein(xf)
for c in clips[1:]:
timeline = timeline.crossfadein(xf).fx(lambda clip: clip)
timeline = CompositeVideoClip([timeline.set_end(timeline.duration), c.set_start(timeline.duration - xf)])


# Ensure duration
final = CompositeVideoClip([clips[0]])
cur = clips[0]
for idx in range(1, len(clips)):
cur = CompositeVideoClip([cur.set_end(cur.duration), clips[idx].set_start(cur.duration - xf)])
final = cur.set_duration(VIDEO_DURATION)


# audio attach
aclip = AudioFileClip(voice_mp3)
final = final.set_audio(aclip)


# render
mp4 = str(OUT / f"signal_long_{uid}.mp4")
final.write_videofile(mp4, fps=30, codec='libx264', audio_codec='aac')


# Upload
n = datetime.utcnow().strftime('%Y-%m-%d')
title = f"SIGNAL LOG — {n} — {random.choice(['The Pulse Beneath','Whisper Archive','Eclipsera'])}"
desc = (
"From the T-3012 archives. God is not gone. He's buried.\n\n"
"Godfinders project — automated transmission loop."
)
tags = ["godfinders","ai","cosmic horror","existential","shorts"]
if os.getenv('CHANNEL_KEYWORDS'):
tags += [t.strip() for t in os.getenv('CHANNEL_KEYWORDS').split(',') if t.strip()]
vid = upload_to_youtube(mp4, title, desc, tags)
print('Uploaded video ID:', vid)




if __name__ == '__main__':
run_pipeline()
