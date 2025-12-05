# Churn Prediction Explainer Video

3blue1brown-style animated explanation with integrated TTS using manim-voiceover.

## What's This?

A complete ~10 minute video explaining the entire churn prediction project:
- What failed (generative model)
- What worked (time-aware encoder)
- The math and intuition
- Honest about mistakes
- 3b1b style narration

## Quick Start

```bash
# Install dependencies
pip install manim-voiceover[gtts]

# Render with voiceover (auto-generates speech!)
manim -pqh churn_prediction_explained.py FullVideo

# Output: media/videos/churn_prediction_explained/1080p60/FullVideo.mp4
```

That's it! The voiceover is generated automatically using Google's free TTS (gTTS).

## Individual Scenes

Render just one part:

```bash
# Just the intro
manim -pqh churn_prediction_explained.py IntroScene

# Just the time encoding explanation
manim -pqh churn_prediction_explained.py TimeEncodingScene

# Just the results
manim -pqh churn_prediction_explained.py ResultsScene
```

All scenes:
- `IntroScene` - What's the problem?
- `ProblemScene` - Event timelines and the 13-day gap
- `GenerativeApproachScene` - Why the first attempt failed
- `TimeEncodingScene` - The log + fourier breakthrough
- `FocalLossScene` - Handling class imbalance
- `ArchitectureScene` - Building the model
- `ResultsScene` - 0.996 AUROC and lessons learned

## Quality Settings

```bash
# Fast preview (480p, 15fps)
manim -pql churn_prediction_explained.py IntroScene

# Medium (720p, 30fps)
manim -pqm churn_prediction_explained.py FullVideo

# High quality (1080p, 60fps) - recommended
manim -pqh churn_prediction_explained.py FullVideo

# 4K (2160p, 60fps)
manim -pqk churn_prediction_explained.py FullVideo
```

## How It Works

Uses `manim-voiceover` to integrate TTS directly into animations:

```python
with self.voiceover(text="so here's the problem...") as tracker:
    self.play(Write(title))
    self.wait(tracker.duration)
```

The library:
- Generates speech with gTTS (free, no API key needed)
- Caches audio files
- Auto-syncs animations with narration
- Creates subtitles automatically

## Customization

### Change Voice

Edit the script to use different TTS engines:

```python
# Default: gTTS (free, no setup)
self.set_speech_service(GTTSService())

# Or use Azure TTS (better quality, needs API key)
from manim_voiceover.services.azure import AzureService
self.set_speech_service(AzureService(
    voice="en-US-AriaNeural",
    style="newscast-casual"
))

# Or use Coqui TTS (local, offline)
from manim_voiceover.services.coqui import CoquiService
self.set_speech_service(CoquiService())
```

### Adjust Timing

Change wait times or animation speeds in the code:

```python
self.wait(2)  # Make shorter: self.wait(1)
self.play(Write(text), run_time=2)  # Faster: run_time=1
```

### Edit Narration

Just change the text in the `voiceover()` calls:

```python
with self.voiceover(text="your new narration here"):
    # animations
```

## Output Files

After rendering:

```
media/videos/churn_prediction_explained/1080p60/
├── FullVideo.mp4              # Final video with audio
└── voiceovers/                # Cached audio files
    ├── intro_scene_xxxxx.mp3
    ├── problem_scene_xxxxx.mp3
    └── ...
```

The voiceover audio is cached, so re-rendering the same scene with unchanged narration is instant.

## Troubleshooting

### "No module named 'manim'"

```bash
pip install manim
```

### "No module named 'manim_voiceover'"

```bash
pip install manim-voiceover[gtts]
```

### LaTeX errors

Manim needs LaTeX for math formulas:

```bash
# Mac
brew install --cask mactex

# Ubuntu
sudo apt install texlive-full

# Windows
# Download and install MiKTeX from miktex.org
```

### Audio not working

Make sure gTTS is installed:

```bash
pip install gTTS
```

### Want better voice quality?

Use Azure TTS instead:

```bash
pip install manim-voiceover[azure]
```

Then set up Azure credentials and change the service in the code.

## Why manim-voiceover?

- **Integrated**: TTS happens during rendering, not separately
- **Synced**: Animations automatically match narration duration
- **Cached**: Re-renders are fast (audio is cached)
- **Flexible**: Easy to swap TTS engines
- **Subtitles**: Auto-generates .srt files
- **Free**: gTTS needs no API keys

vs the old way:
1. Render silent video
2. Generate audio separately
3. Manually sync timing
4. Combine with ffmpeg
5. Add subtitles manually

## Tips

1. **Test quickly**: Use `-pql` for fast previews while developing
2. **Render sections**: Test individual scenes before full video
3. **Cache works**: Don't delete `media/videos/.../voiceovers/` folder
4. **Edit iteratively**: Change animations, keep narration cached
5. **Export subtitles**: Look in the voiceovers folder for .srt files

## Example Output

The full video is ~10 minutes covering:
- 0:00-0:20: Introduction
- 0:20-1:10: The problem
- 1:10-2:30: Failed generative approach
- 2:30-4:30: Time encoding solution
- 4:30-6:00: Focal loss
- 6:00-7:30: Architecture
- 7:30-10:00: Results and lessons

All with synchronized narration and 3blue1brown-style animations.

## License

Explains the churn prediction project. Adapt for your own use.
