import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import json
import random
import matplotlib.pyplot as plt

from model import NeuralNavigator

def load_vocab(vocab_file):
    with open(vocab_file, 'r') as f:
        return json.load(f)

def process_text(text, vocab, max_len=10):
    text = text.lower().replace('.', '').split()
    text_indices = [vocab.get(word, vocab.get("<UNK>", 3)) for word in text]
    text_indices = [vocab.get("<SOS>", 1)] + text_indices + [vocab.get("<EOS>", 2)]
    
    if len(text_indices) < max_len:
        text_indices += [vocab.get("<PAD>", 0)] * (max_len - len(text_indices))
    else:
        text_indices = text_indices[:max_len]
        
    return torch.tensor(text_indices, dtype=torch.long).unsqueeze(0) # (1, SeqLen)

def predict_and_visualize():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'neural_navigator.pth'
    VOCAB_PATH = 'vocab.json'
    TEST_DATA_DIR = 'test_data'
    OUTPUT_DIR = 'predictions'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Load Vocab
    vocab = load_vocab(VOCAB_PATH)
    
    # Load Model
    model = NeuralNavigator(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Get Test Files
    annotations_dir = os.path.join(TEST_DATA_DIR, 'annotations')
    images_dir = os.path.join(TEST_DATA_DIR, 'images')
    ann_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.json')])
    
    # Select 5 random samples
    samples = random.sample(ann_files, 5)
    
    for i, ann_file in enumerate(samples):
        with open(os.path.join(annotations_dir, ann_file), 'r') as f:
            data = json.load(f)
            
        img_path = os.path.join(images_dir, data['image_file'])
        original_image = Image.open(img_path).convert('RGB')
        
        # Prepare Input
        image_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
        text_tensor = process_text(data['text'], vocab).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            predicted_path = model(image_tensor, text_tensor)
            
        # Post-process Path
        # Output is (1, 10, 2) in normalized coords [0, 1] (if we normalized by 128)
        # Wait, in data_loader we normalized by dividing by 128.
        # So we need to multiply by 128.
        path_coords = predicted_path.squeeze(0).cpu().numpy() * 128.0
        
        # Draw
        draw = ImageDraw.Draw(original_image)
        
        # Draw path points
        # Draw lines between points
        for j in range(len(path_coords) - 1):
            x1, y1 = path_coords[j]
            x2, y2 = path_coords[j+1]
            draw.line((x1, y1, x2, y2), fill='blue', width=2)
            draw.ellipse((x1-2, y1-2, x1+2, y1+2), fill='blue')
            
        # Draw last point
        last_x, last_y = path_coords[-1]
        draw.ellipse((last_x-3, last_y-3, last_x+3, last_y+3), fill='red') # Target
        
        # Save
        save_path = os.path.join(OUTPUT_DIR, f"pred_{data['image_file']}")
        original_image.save(save_path)
        print(f"Saved prediction to {save_path}")

if __name__ == '__main__':
    predict_and_visualize()
