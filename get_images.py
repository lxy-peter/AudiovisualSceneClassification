import cv2
import os


audio_path = 'audiosetdl/dataset/data/eval_segments/audio'
video_path = 'audiosetdl/dataset/data/eval_segments/video'
image_path = 'audiosetdl/dataset/data/eval_segments/image'
os.mkdir(image_path)


audio_file_ids = [file.replace('.flac', '') for file in os.listdir(audio_path) if not file.startswith('.')]
video_file_ids = [file.replace('.flac', '') for file in os.listdir(audio_path) if not file.startswith('.')]
file_ids = set(audio_file_ids).intersection(set(video_file_ids))

for n, file_id in enumerate(file_ids):
    if n % 10 == 0:
        print('Processing video', n+1, '/', len(file_ids))
    
    audio_file_path = audio_path + '/' + file_id + '.flac'
    video_file_path = video_path + '/' + file_id + '.mp4'
    video = cv2.VideoCapture(video_file_path)
    
    imgs = []
    while True:
        ret, img = video.read()
        if ret:
            imgs.append(img)
        else:
            break
    
    N = len(imgs)
    if N < 3:
        continue
    
    for i, pos in enumerate([0.3, 0.5, 0.7]):
        img = imgs[int(pos * N)]
        img_filename = image_path + '/' + file_id + '_' + str(i) + '.jpg'
        cv2.imwrite(img_filename, img)
        


