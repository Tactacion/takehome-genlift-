# Quick Start - Video Production

## TL;DR

```bash
# Install (one time)
pip install manim-voiceover[gtts]

# Render video with voice (automatic!)
manim -pqh churn_prediction_explained.py FullVideo
```

Done. Video with voiceover is at `media/videos/churn_prediction_explained/1080p60/FullVideo.mp4`

## What Changed?

Now using `manim-voiceover` - way simpler than before:

**Old approach:**
1. Render silent video
2. Run separate TTS script
3. Manually combine video + audio
4. Generate subtitles separately
5. Hope timing matches

**New approach:**
1. Just render

That's it. TTS is integrated into the animation code.

## How It Works

The code looks like this:

```python
with self.voiceover(text="so here's the problem...") as tracker:
    self.play(Write(title))
    self.wait(tracker.duration)
```

- Narration is embedded in the animation
- Audio is auto-generated with gTTS (free, no API key)
- Timing is automatic
- Subtitles created automatically
- Audio is cached (re-renders are instant)

## Test It

```bash
bash test_render.sh
```

This renders just the intro scene (20 seconds) in low quality for quick preview.

## Individual Scenes

```bash
# Time encoding explanation (the most important part)
manim -pqh churn_prediction_explained.py TimeEncodingScene

# Results reveal
manim -pqh churn_prediction_explained.py ResultsScene

# Failed approach (honesty!)
manim -pqh churn_prediction_explained.py GenerativeApproachScene
```

## Full Video

```bash
# High quality (recommended)
manim -pqh churn_prediction_explained.py FullVideo

# 4K (if you want)
manim -pqk churn_prediction_explained.py FullVideo
```

~10 minute video with full narration and subtitles.

## Benefits

✓ Free TTS (no API keys)
✓ Auto-synced timing
✓ Cached audio (fast re-renders)
✓ Auto-generated subtitles
✓ Easy to edit narration
✓ No ffmpeg needed
✓ No manual timing

## Files

- `churn_prediction_explained.py` - Complete animation with embedded narration
- `test_render.sh` - Quick test script
- `requirements.txt` - Just 2 dependencies
- `README_VIDEO.md` - Full documentation

## Output Location

After rendering:

```
media/videos/churn_prediction_explained/1080p60/
├── FullVideo.mp4      # Final video with audio
└── FullVideo.srt      # Subtitles

media/voiceovers/
└── *.mp3              # Cached audio files
```

## Edit Narration

Just change the text in the code:

```python
with self.voiceover(text="new narration here"):
    # animations
```

Re-render and the audio is auto-regenerated.

## Want Better Voice?

Change from gTTS to Azure TTS:

```bash
pip install manim-voiceover[azure]
```

Then in the code:

```python
from manim_voiceover.services.azure import AzureService
self.set_speech_service(AzureService(
    voice="en-US-AriaNeural",
    style="newscast-casual"
))
```

Needs Azure credentials but sounds way better.

## That's It

Just run `manim -pqh churn_prediction_explained.py FullVideo` and you get a complete 10-minute video explaining the entire project with voiceover and subtitles.

Way simpler than the old approach!
