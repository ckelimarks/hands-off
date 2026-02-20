# Face Touch Detector

A real-time webcam-based detector that alerts you when your hands touch your face for more than 1 second. Helps break unconscious habits like beard pulling or face touching.

## Features

- Real-time hand and face detection using MediaPipe
- **ANNOYING ALARM SOUND** when hands touch face for > 1 second
- Visual red screen alert overlay
- Adjustable sensitivity and timing thresholds
- Console logging of touch events
- Mirror-mode webcam display

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python detector.py
```

### Controls

- **q** - Quit the application

### How It Works

1. Detects your face using MediaPipe Face Detection
2. Tracks both hands using MediaPipe Hands
3. Calculates proximity between hands and face
4. Starts a timer when hands enter face area
5. Shows a red alert overlay when contact exceeds 1 second
6. **Plays a continuous, annoying alarm sound** (alternating high-pitched beeps)
7. Sound stops immediately when you remove your hands
8. Logs all touch events to console with timestamps

### Customization

Edit the `detector.py` file to adjust:

```python
detector = FaceTouchDetector(
    touch_threshold_seconds=1.0,   # Alert after this many seconds
    proximity_threshold=0.08,      # How close counts as "touching" (0-1)
    enable_sound=True              # Set to False to disable alarm sound
)
```

### Sound Customization

The alarm uses alternating 1200Hz and 800Hz tones with amplitude modulation for maximum annoyance. To adjust:
- Edit `_generate_annoying_alarm()` in detector.py:31
- Change `freq1` and `freq2` for different pitches
- Adjust `duration` for longer/shorter beeps

### Tips

- Position yourself so your face is clearly visible in the webcam
- Ensure good lighting for better detection accuracy
- The proximity threshold determines how close your hands need to be - increase for stricter detection
- Keep the window visible while working to see alerts

## Technical Details

- Uses MediaPipe for hand and face landmark detection
- Tracks 21 hand landmarks per hand (2 hands max)
- Face detection with bounding box
- Euclidean distance calculation for proximity
- Real-time processing at webcam framerate

## Next Steps

Potential enhancements:
- Statistics tracking (daily touch count, duration)
- System tray integration for background monitoring
- Multiple alarm sound options (siren, buzzer, voice alert)
- Focus on specific face regions (chin/beard area only)
- Haptic feedback via phone notification
