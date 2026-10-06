# LinguaDub Studio

AI video dubbing with voice cloning, script editing, line timing control,
and translations that fit each line's time slot.

Pipeline: Whisper (transcribe) → MyMemory / NLLB / Google (translate) → XTTS v2 (voice clone).

## Requirements
- Python 3.10–3.12
- ffmpeg
- NVIDIA GPU recommended (CPU works but is very slow)
- About 2.4 GB of disk for NLLB on first use, plus the XTTS and Whisper models

## Run in Google Colab (easiest)
Set the runtime to GPU, then run:
```python
!apt-get -qq install ffmpeg
!git clone https://github.com/digimarketingai/LinguaDub.git
%cd LinguaDub
!pip install -q -r requirements.txt
!python app.py
```

## Run locally
```bash
# install ffmpeg first (Ubuntu: sudo apt install ffmpeg | macOS: brew install ffmpeg)
git clone https://github.com/digimarketingai/full-code.git
cd full-code
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Then open the local URL printed in the terminal.

## Notes
- The first run downloads the models, so it will be slow.
- XTTS v2 is released under the Coqui Public Model License (non-commercial). Check it before commercial use.
- Only clone voices you have permission to use.
