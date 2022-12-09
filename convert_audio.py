from pydub import AudioSegment
import os


audio_dir = 'audiosetdl/dataset/data/eval_segments/audio'
audio_files = [file for file in os.listdir(audio_dir) if not file.startswith('.')]

for i, file in enumerate(audio_files):
    if i % 100 == 0:
        print('Processing audio file', i+1, '/', len(audio_files))
    
    full_path = audio_dir + '/' + file
    audio = AudioSegment.from_file(full_path, format='flac', frame_rate=48000, channels=2)
    audio = audio.set_frame_rate(22000)
    audio = audio.set_channels(1)
    
    new_path = full_path.replace('/audio/', '/audio_wav/').replace('.flac', '.wav')
    audio.export(new_path, format='wav')
    
    
    
    