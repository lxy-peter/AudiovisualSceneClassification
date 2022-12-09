# Audiovisual Classification of General Scenes

## Instructions to Run

1. Please navigate to be inside the folder of this repository using `cd`, then execute the following commands:
```
mkdir image_model multi_model sound_model
git clone https://github.com/speedyseal/audiosetdl
cd audiosetdl
```
2. Follow the instructions at https://github.com/speedyseal/audiosetdl to install dependencies. Then, run:
```
python download_audioset.py dataset
```
  A common error is locating `ffmpeg` and/or `ffprobe`. You may have to edit `download_audioset.py` and specify the correct default value for argument corresponding to paths of these executables.
