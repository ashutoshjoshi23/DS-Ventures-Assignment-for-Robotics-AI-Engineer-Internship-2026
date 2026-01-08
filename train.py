import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from data_loader import get_dataloader, NeuralNavigatorDataset
from model import NeuralNavigator

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    DATA_DIR = 'data'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('log.txt', 'w') as f:
        f.write(f"Using device: {DEVICE}\n")
    
    print(f"Using device: {DEVICE}")
    
    # Data Loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = NeuralNavigatorDataset(DATA_DIR, transform=transform)
    vocab = full_dataset.vocab
    print(f"Vocabulary size: {len(vocab)}")
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=None)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=None)
    
    # Model
    model = NeuralNavigator(vocab_size=len(vocab)).to(DEVICE)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = []
    val_loss_history = []
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, texts, targets in progress_bar:
            images = images.to(DEVICE)
            texts = texts.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward
            outputs = model(images, texts)
            
            # Loss
            loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, texts, targets in val_loader:
                images = images.to(DEVICE)
                texts = texts.to(DEVICE)
                targets = targets.to(DEVICE)
                
                outputs = model(images, texts)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
    # Save Model
    torch.save(model.state_dict(), 'neural_navigator.pth')
    # Save vocab as well
    import json
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    print("Model and vocab saved.")
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig('training_loss.png')
    print("Loss curve saved to training_loss.png")

if __name__ == '__main__':
    train()
