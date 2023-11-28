TTS engine pushed in a Python package with easy inferencing functions.

Based on fork from https://github.com/yl4579/StyleTTS2 by Aaron (Yinghao) Li.

# How to install: 
- `sudo apt-get install espeak-ng git`
- `git lfs install`
- `pip install .`
The model weights will be automatically downloaded during the first run. 


# Run with:
```python
from styletts2_inferencer import StyleTTS2Inferencer

tts = StyleTTS2Inferencer() # defaults to using cuda/gpu
tts.tts_to_file(
    text="Hello! How are you?", # automatically switches to long form generation for multiline texts.
    filepath="test.wav" # should be an absolute path
)
# Output: 
# RTF = 0.410011
# Written output to file test.wav
```
See test.wav for an example output.



# Improvements to make
- [ ] Add parameter for speech expresseviness and number of diffusion steps as shown in [the demo](https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LJSpeech.ipynb)

