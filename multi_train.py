from image_train import ImageClassifier
import os
from PIL import Image
from sound_train import SoundClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
from transformers import ViTImageProcessor



device = "cuda" if torch.cuda.is_available() else "cpu"



f = open('data/classidx_to_classid.txt', 'r')
classes = [line for line in f.read().split('\n') if line != '']
f.close()
N_classes = len(classes)



class MultimodalDataset(Dataset):
    
    def __init__(self, split):
        assert split == 'train' or split == 'dev' or split == 'test'
        
        # Image files
        
        image_dir = 'audiosetdl/dataset/data/eval_segments/image'
        fnames_tmp = {file for file in os.listdir(image_dir) if not file.startswith('.')}
        
        f = open('audiosetdl/dataset/data/eval_segments/' + split + '.txt', 'r')
        wanted_sampleids = [line for line in f.read().split('\n') if line != '']
        f.close()
        
        self.image_fnames = []
        self.imgfname_to_sampleid = {}
        for fname in fnames_tmp:
            pos = fname.rfind('_')
            sampleid = fname[:pos]
            pos = sampleid.rfind('_')
            sampleid = sampleid[:pos]
            pos = sampleid.rfind('_')
            sampleid = sampleid[:pos]
            if sampleid in wanted_sampleids:
                self.imgfname_to_sampleid[fname] = sampleid
                self.image_fnames.append(fname)
                
        self.sampleid_to_imgfname = {}
        for fname in self.imgfname_to_sampleid:
            sampleid = self.imgfname_to_sampleid[fname]
            if sampleid in self.sampleid_to_imgfname:
                self.sampleid_to_imgfname[sampleid].append(fname)
            else:
                self.sampleid_to_imgfname[sampleid] = [fname]
        
        
        # Audio files
        
        self.std_audio_samples = 220000
        audio_dir = 'audiosetdl/dataset/data/eval_segments/audio_wav'
        
        fnames_tmp = {file for file in os.listdir(audio_dir) if not file.startswith('.')}
        
        self.audio_fnames = []
        self.audfname_to_sampleid = {}
        for filename in fnames_tmp:
            pos = filename.rfind('_')
            sampleid = filename[:pos]
            pos = sampleid.rfind('_')
            sampleid = sampleid[:pos]
            if sampleid in wanted_sampleids:
                self.audfname_to_sampleid[filename] = sampleid
                self.audio_fnames.append(filename)
            
        self.sampleid_to_audfname = {self.audfname_to_sampleid[filename] : filename for filename in self.audfname_to_sampleid}
        
        
        
        f = open('data/sampleid_to_classidx.txt', 'r')
        lines = [line for line in f.read().split('\n') if line != '']
        f.close()
        
        self.examples = []
        feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        
        for i, line in enumerate(lines):
            space_pos = line.find(' ')
            sampleid = line[:space_pos]
            
            if sampleid in self.sampleid_to_imgfname:
                
                imgfnames = self.sampleid_to_imgfname[sampleid]
                audfname = self.sampleid_to_audfname[sampleid]
                
                classidxs = [int(num) for num in line[space_pos+1:].split()]
                
                
                
                audio, sr = torchaudio.load(audio_dir + '/' + audfname)
                assert sr == 22000
                audio = torch.squeeze(audio)
                assert len(audio.shape) == 1
                
                # print(torch.max(torch.abs(waveform)))
                # audio *= 256
                
                if audio.shape[0] < self.std_audio_samples:
                    diff = self.std_audio_samples - audio.shape[0]
                    left = int(diff / 2)
                    right = diff - left
                    audio = F.pad(audio, (left, right), 'constant', 0)
                elif audio.shape[0] > self.std_audio_samples:
                    diff = audio.shape[0] - self.std_audio_samples
                    left = int(diff / 2)
                    right = diff - left
                    audio = audio[left:-right]
                
                assert len(audio.shape) == 1 and audio.shape[0] == self.std_audio_samples
                
                audio = torch.unsqueeze(audio, 1)
                audio = torch.unsqueeze(audio, 0)
                
                
                
                for imgfname in imgfnames:
                    image = Image.open(image_dir + '/' + imgfname)
                    image = feature_extractor(image, return_tensors='pt')['pixel_values']
                    image = torch.squeeze(image)
                    
                    if split == 'train':
                        label_probs = torch.zeros(N_classes)
                        label_probs[classidxs] = 1
                        label_probs = label_probs / torch.sum(label_probs)
                        
                        self.examples.append((image, audio, label_probs))
                    
                    else:
                        self.examples.append((image, audio, classidxs))
    
    
    def __len__(self):
        return len(self.examples)


    def __getitem__(self, idx):
        example = self.examples[idx]
        return {'Input': (example[0], example[1]), 'Label': example[2]}
                    
    

class MultimodalClassifier(nn.Module):

    def __init__(self):
        
        super(MultimodalClassifier, self).__init__()
        
        self.image_classifier = ImageClassifier()
        path = 'image_model/epoch15.pt'
        if device == 'cpu':
            self.image_classifier.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif 'cuda' in device:
            self.image_classifier.load_state_dict(torch.load(path))
        
        self.sound_classifier = SoundClassifier()
        path = 'sound_model/epoch15.pt'
        if device == 'cpu':
            self.sound_classifier.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif 'cuda' in device:
            self.sound_classifier.load_state_dict(torch.load(path))
        
        self.aud_linear = nn.Linear(1024, N_classes)
        
        self.linear = nn.Linear(20, N_classes)
    
    def forward(self, img, aud):
        
        img = self.image_classifier(img)

        aud = self.sound_classifier.soundnet_feats(aud)
        aud = self.aud_linear(aud)
        aud = F.relu(aud)
        
        if len(aud.shape) == 1:
            aud = torch.unsqueeze(aud, dim=0)
        
        x = torch.hstack((img, aud))
        x = self.linear(x)
        
        return x




def train(load_model=None):
    batch_size = 4
    
    model = MultimodalClassifier()
    print(model)
    
    if device == 'cpu' and load_model is not None:
        model.load_state_dict(torch.load(load_model, map_location=torch.device('cpu')))
    elif 'cuda' in device and load_model is not None:
        model.load_state_dict(torch.load(load_model))
    print('Beginning training from', load_model)
    
    model.eval()
    
    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    
    train_data = MultimodalDataset(split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    epochs = 25
    for epoch in range(15, epochs):
        epoch_loss = 0
        data_count = 0
        
        
        if epoch == 0:
            print('Freezing image and sound classifier parameters to initialize linear layer weights')
            for param in model.image_classifier.parameters():
                param.requires_grad = False
            for param in model.sound_classifier.parameters():
                param.requires_grad = False
        
        
        for i, data in enumerate(train_loader, 0):
            
            image = data['Input'][0].to(device)
            audio = data['Input'][1].to(device)
            labels = data['Label'].to(device)
            
            data_count += batch_size
            
            optimizer.zero_grad()
            outputs = model.forward(image, audio)
            
            loss = criterion(outputs, labels)
            
            if torch.any(torch.isnan(loss)):
                raise RuntimeError
        
            epoch_loss += float(loss)
            
            if i % 5 == 0:
                print("Epoch", epoch+1,
                      "| Processing data", data_count, "/", len(train_data),
                      "| Loss/sample:", float(loss) / batch_size)
                if i % 1000 == 0:
                    torch.save(model.state_dict(), "multi_model/model.pt")
            
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), 'multi_model/epoch' + str(epoch+1) + '.pt')
        print('')
        print('**************************************************************')
        print('Epoch ' + str(epoch+1) + ' loss:', epoch_loss)
        print('**************************************************************')
        print('')
        
        
        if epoch == 4:
            print('Unfreezing all parameters')
            for param in model.parameters():
                param.requires_grad = True
        

if __name__ == '__main__':
    train('multi_model/epoch25.pt')

