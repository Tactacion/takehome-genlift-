#!/bin/bash

# Quick test render with voiceover
# Run with: bash test_render.sh

echo "Testing manim-voiceover installation..."
python -c "from manim_voiceover.services.gtts import GTTSService; print('✓ manim-voiceover with gTTS ready')" || {
    echo "manim-voiceover not installed. Run: pip install manim-voiceover[gtts]"
    exit 1
}

echo ""
echo "Rendering intro scene (low quality for quick preview)..."
manim -pql churn_prediction_explained.py IntroScene

echo ""
echo "If that worked, try:"
echo "  - High quality intro: manim -pqh churn_prediction_explained.py IntroScene"
echo "  - Full video: manim -pqh churn_prediction_explained.py FullVideo"
echo ""
echo "Video saved to: media/videos/churn_prediction_explained/480p15/IntroScene.mp4"
echo "Audio cached in: media/videos/churn_prediction_explained/480p15/voiceovers/"
