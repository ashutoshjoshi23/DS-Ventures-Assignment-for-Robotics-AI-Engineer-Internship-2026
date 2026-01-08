import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class NeuralNavigatorDataset(Dataset):
    def __init__(self, data_dir, transform=None, vocab=None, max_text_len=10):
        self.data_dir = data_dir
        self.transform = transform
        self.max_text_len = max_text_len
        self.images_dir = os.path.join(data_dir, 'images')
        self.annotations_dir = os.path.join(data_dir, 'annotations')
        
        self.annotation_files = sorted([f for f in os.listdir(self.annotations_dir) if f.endswith('.json')])
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
            
    def _build_vocab(self):
        vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        idx = 4
        for ann_file in self.annotation_files:
            with open(os.path.join(self.annotations_dir, ann_file), 'r') as f:
                data = json.load(f)
                text = data['text'].lower().replace('.', '').split()
                for word in text:
                    if word not in vocab:
                        vocab[word] = idx
                        idx += 1
        return vocab

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        ann_file = self.annotation_files[idx]
        with open(os.path.join(self.annotations_dir, ann_file), 'r') as f:
            data = json.load(f)
            
        # Load Image
        img_path = os.path.join(self.images_dir, data['image_file'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Process Text
        text = data['text'].lower().replace('.', '').split()
        text_indices = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text]
        text_indices = [self.vocab["<SOS>"]] + text_indices + [self.vocab["<EOS>"]]
        
        # Padding
        if len(text_indices) < self.max_text_len:
            text_indices += [self.vocab["<PAD>"]] * (self.max_text_len - len(text_indices))
        else:
            text_indices = text_indices[:self.max_text_len]
            
        text_tensor = torch.tensor(text_indices, dtype=torch.long)
        
        # Process Path (Target)
        # If it's test data, path might not exist or we might not need it for inference, 
        # but for training we do. The assignment says test annotations have 'target' but maybe not 'path'?
        # Let's check if 'path' exists.
        if 'path' in data:
            path = data['path']
            # Normalize coordinates to [0, 1]
            path_tensor = torch.tensor(path, dtype=torch.float32) / 128.0
            # Flatten to (10, 2) -> (20,) if needed, or keep as (10, 2)
            # The model output will likely be (10, 2)
        else:
            # For test set if path is missing, return dummy
            path_tensor = torch.zeros((10, 2), dtype=torch.float32)

        return image, text_tensor, path_tensor

def get_dataloader(data_dir, batch_size=32, shuffle=True, vocab=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize with mean and std of ImageNet or just 0.5
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    
    dataset = NeuralNavigatorDataset(data_dir, transform=transform, vocab=vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.vocab
