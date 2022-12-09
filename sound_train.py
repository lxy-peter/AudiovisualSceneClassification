import os
from soundnet import SoundNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio


device = "cuda" if torch.cuda.is_available() else "cpu"


f = open('data/classidx_to_classid.txt', 'r')
classes = [line for line in f.read().split('\n') if line != '']
f.close()
N_classes = len(classes)



class SoundDataset(Dataset):
    
    def __init__(self, split):
        assert split == 'train' or split == 'dev' or split == 'test'
        
        self.std_audio_samples = 220000
        audio_dir = 'audiosetdl/dataset/data/eval_segments/audio_wav'
        
        self.filenames_tmp = {file for file in os.listdir(audio_dir) if not file.startswith('.')}
        
        f = open('audiosetdl/dataset/data/eval_segments/' + split + '.txt', 'r')
        wanted_sampleids = [line for line in f.read().split('\n') if line != '']
        f.close()
        
        self.filenames = []
        self.filename_to_sampleid = {}
        for filename in self.filenames_tmp:
            pos = filename.rfind('_')
            sampleid = filename[:pos]
            pos = sampleid.rfind('_')
            sampleid = sampleid[:pos]
            if sampleid in wanted_sampleids:
                self.filename_to_sampleid[filename] = sampleid
                self.filenames.append(filename)
            
        self.sampleid_to_filename = {self.filename_to_sampleid[filename] : filename for filename in self.filename_to_sampleid}

        f = open('data/sampleid_to_classidx.txt', 'r')
        lines = [line for line in f.read().split('\n') if line != '']
        f.close()
        
        self.examples = []
        
        for i, line in enumerate(lines):
            space_pos = line.find(' ')
            sampleid = line[:space_pos]
            
            if sampleid in self.sampleid_to_filename:
                classidxs = [int(num) for num in line[space_pos+1:].split()]
                
                waveform, sr = torchaudio.load(audio_dir + '/' + \
                                               self.sampleid_to_filename[sampleid])
                assert sr == 22000
                waveform = torch.squeeze(waveform)
                assert len(waveform.shape) == 1
                
                # print(torch.max(torch.abs(waveform)))
                waveform *= 256
                
                if waveform.shape[0] < self.std_audio_samples:
                    diff = self.std_audio_samples - waveform.shape[0]
                    left = int(diff / 2)
                    right = diff - left
                    waveform = F.pad(waveform, (left, right), 'constant', 0)
                elif waveform.shape[0] > self.std_audio_samples:
                    diff = waveform.shape[0] - self.std_audio_samples
                    left = int(diff / 2)
                    right = diff - left
                    waveform = waveform[left:-right]
                
                assert len(waveform.shape) == 1 and waveform.shape[0] == self.std_audio_samples
                
                waveform = torch.unsqueeze(waveform, 1)
                waveform = torch.unsqueeze(waveform, 0)
                
                if split == 'train':
                    label_probs = torch.zeros(N_classes)
                    label_probs[classidxs] = 1
                    label_probs = label_probs / torch.sum(label_probs)
                    
                    self.examples.append((waveform, label_probs))
                    
                else:
                    self.examples.append((waveform, classidxs))
                    
        

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {'Input': example[0], 'Label': example[1]}


class SoundClassifier(nn.Module):
    
    def __init__(self):
        super(SoundClassifier, self).__init__()

        self.soundnet = SoundNet()
        
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, N_classes)
    
    def forward(self, x):
        
        x = self.soundnet(x)
        x = torch.mean(x, dim=2)
        x = torch.squeeze(x)
        
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        
        return x
    
    def soundnet_feats(self, x):
        
        x = self.soundnet(x)
        x = torch.mean(x, dim=2)
        x = torch.squeeze(x)
        
        return x



def train(load_model=None):
    batch_size = 16
    
    model = SoundClassifier()
    print(model)
    
    if load_model:
        if device == 'cpu':
            model.load_state_dict(torch.load(load_model, map_location=torch.device('cpu')))
        elif 'cuda' in device:
            model.load_state_dict(torch.load(load_model))
    else:
        model.soundnet.load_state_dict(torch.load('soundnet8_final.pth'))
    print('Beginning training from', load_model)
    
    model.eval()
    
    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    
    train_data = SoundDataset(split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    epochs = 15
    for epoch in range(epochs):
        epoch_loss = 0
        data_count = 0
        
        if epoch == 0:
            print('Freezing SoundNet parameters to initialize linear layers')
            for param in model.soundnet.parameters():
                param.requires_grad = False
            print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
        for i, data in enumerate(train_loader, 0):
            
            inputs = data['Input'].to(device)
            labels = data['Label'].to(device)
            
            data_count += batch_size
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            
            loss = criterion(outputs, labels)
            
            if torch.any(torch.isnan(loss)):
                raise RuntimeError
        
            epoch_loss += float(loss)
            
            if i % 5 == 0:
                print("Epoch", epoch+1,
                      "| Processing data", data_count, "/", len(train_data),
                      "| Loss/sample:", float(loss) / batch_size)
                if i % 1000 == 0:
                    torch.save(model.state_dict(), "sound_model/model.pt")
            
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), 'sound_model/epoch' + str(epoch+1) + '.pt')
        print('')
        print('**************************************************************')
        print('Epoch ' + str(epoch+1) + ' loss:', epoch_loss)
        print('**************************************************************')
        print('')
        
        if epoch == 4:
            print('Unfreezing all parameters')
            for param in model.soundnet.parameters():
                param.requires_grad = True
            print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':
    train()


