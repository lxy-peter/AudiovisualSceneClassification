import os
import random


sampleids = list(set([file[:-6] for file in os.listdir('audiosetdl/dataset/data/eval_segments/image') if file != '' and not file.startswith('.')]))
random.shuffle(sampleids)

dev_begin_idx = int(len(sampleids) * 0.8)
test_begin_idx = int(len(sampleids) * 0.9)

train = sampleids[:dev_begin_idx]
dev = sampleids[dev_begin_idx:test_begin_idx]
test = sampleids[test_begin_idx:]

f = open('audiosetdl/dataset/data/eval_segments/orig/train.txt', 'w')
for sampleid in train:
    f.write(sampleid + '\n')
f.close()

f = open('audiosetdl/dataset/data/eval_segments/orig/dev.txt', 'w')
for sampleid in dev:
    f.write(sampleid + '\n')
f.close()

f = open('audiosetdl/dataset/data/eval_segments/orig/test.txt', 'w')
for sampleid in test:
    f.write(sampleid + '\n')
f.close()




