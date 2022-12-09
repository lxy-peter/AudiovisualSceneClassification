from multi_train import MultimodalClassifier, MultimodalDataset
import torch
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"

    

def predict(model_path):
    model = MultimodalClassifier()
    if torch.cuda.is_available() and 'cuda' in device:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    
    dataset = MultimodalDataset(split='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    top1_correct = 0
    top2_correct = 0

    for i, data in enumerate(loader, 0):
        if i % 10 == 0:
            print('Processing sample', i+1, '/', len(dataset))

        image = data['Input'][0].to(device)
        audio = data['Input'][1].to(device)
        labels = data['Label']
        
        outputs = model.forward(image, audio)
        outputs = torch.squeeze(outputs)
        
        
        _, top1 = torch.max(outputs, dim=0)
        _, top2 = torch.topk(outputs, k=2, dim=0)
        
        labels = [int(label[0]) for label in labels]
        
        if int(top1) in labels:
            top1_correct += 1
        
        for pred in top2:
            if int(pred) in labels:
                top2_correct += 1
                break
    
    top1_result = 'Top-1 accuracy:' + str(top1_correct / len(dataset))
    top2_result = 'Top-2 accuracy:' + str(top2_correct / len(dataset))

    log_path = model_path.replace('.pt', '.log')
    f = open(log_path, 'w')
    f.write(top1_result + '\n' + top2_result + '\n')
    f.close()
    
    print(log_path + ':')
    print(top1_result)
    print(top2_result)


if __name__ == '__main__':
    predict('multi_model/best.pt')
                
                
                