# Audiovisual Classification of General Scenes

## Setup

1. Please navigate to be inside the folder of this repository using `cd`, then execute the following commands:
```
mkdir image_model multi_model sound_model
git clone https://github.com/speedyseal/audiosetdl
cd audiosetdl
```
2. Follow the instructions at https://github.com/speedyseal/audiosetdl to install dependencies. The next step is to run `download_audioset.py`, but a common error is for the program to locate `ffmpeg` and/or `ffprobe`. To run successfully, you may have to edit `download_audioset.py` and specify the correct default value for argument corresponding to paths of these executables. Then, proceed with:
```
python download_audioset.py dataset
```
3. Run:
```
mkdir dataset/data/eval_segments/image
mkdir dataset/data/eval_segments/audio_wav
cd ..
mv tmp/* audiosetdl/dataset/data/eval_segments/
python get_images.py
python convert_audio.py
```
4. Download the file <a href="https://drive.google.com/file/d/12TgiB7jZ00MMoAC7BBOENPvMpRhoPYlz/view?usp=share_link">here</a> and place it in `image_model/`.

## Training


## Inference

To use the image model, multimodal model, or sound model to predict on the test, simply run `image_predict.py`, `multi_predict.py`, or `sound_predict.py`, respectively.

To predict on the validation set, edit the Python file corresponding to the model you wish to run, and change `'test'` to `'dev``.
