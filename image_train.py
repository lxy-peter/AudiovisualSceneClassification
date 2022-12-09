import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessor, ViTForImageClassification



device = "cuda" if torch.cuda.is_available() else "cpu"



f = open('data/classidx_to_classid.txt', 'r')
classes = [line for line in f.read().split('\n') if line != '']
f.close()
N_classes = len(classes)



class ImageDataset(Dataset):
    
    def __init__(self, split):
        assert split == 'train' or split == 'dev' or split == 'test'
        
        image_dir = 'audiosetdl/dataset/data/eval_segments/image'
        self.filenames_tmp = {file for file in os.listdir(image_dir) if not file.startswith('.')}
        
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
            pos = sampleid.rfind('_')
            sampleid = sampleid[:pos]
            if sampleid in wanted_sampleids:
                self.filename_to_sampleid[filename] = sampleid
                self.filenames.append(filename)
                
        self.sampleid_to_filename = {}
        for filename in self.filename_to_sampleid:
            sampleid = self.filename_to_sampleid[filename]
            if sampleid in self.sampleid_to_filename:
                self.sampleid_to_filename[sampleid].append(filename)
            else:
                self.sampleid_to_filename[sampleid] = [filename]
        
        f = open('data/sampleid_to_classidx.txt', 'r')
        lines = [line for line in f.read().split('\n') if line != '']
        f.close()
        
        self.examples = []
        feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        
        for i, line in enumerate(lines):
            space_pos = line.find(' ')
            sampleid = line[:space_pos]
            
            if sampleid in self.sampleid_to_filename:
                
                filenames = self.sampleid_to_filename[sampleid]
                classidxs = [int(num) for num in line[space_pos+1:].split()]
                
                for filename in filenames:
                    image = Image.open(image_dir + '/' + filename)
                    image = feature_extractor(image, return_tensors='pt')['pixel_values']
                    image = torch.squeeze(image)
                    
                    if split == 'train':
                        label_probs = torch.zeros(N_classes)
                        label_probs[classidxs] = 1
                        label_probs = label_probs / torch.sum(label_probs)
                        
                        self.examples.append((image, label_probs))
                    
                    else:
                        self.examples.append((image, classidxs))
    
    
    def __len__(self):
        return len(self.examples)


    def __getitem__(self, idx):
        example = self.examples[idx]
        return {'Input': example[0], 'Label': example[1]}
                    
    

class ImageClassifier(nn.Module):

    def __init__(self):
        
        super(ImageClassifier, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=N_classes, ignore_mismatched_sizes=True)
        
    
    def forward(self, x):
        x = self.vit(x).logits
        
        return x




def train(load_model=None):
    batch_size = 4
    
    model = ImageClassifier()
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
    
    train_data = ImageDataset(split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    epochs = 25
    for epoch in range(epochs):
        epoch_loss = 0
        data_count = 0
        
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
                    torch.save(model.state_dict(), "image_model/model.pt")
            
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), 'image_model/epoch' + str(epoch+1) + '.pt')
        print('')
        print('**************************************************************')
        print('Epoch ' + str(epoch+1) + ' loss:', epoch_loss)
        print('**************************************************************')
        print('')
        

if __name__ == '__main__':
    train('image_model/epoch15.pt')

