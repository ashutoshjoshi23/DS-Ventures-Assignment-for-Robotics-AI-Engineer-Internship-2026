import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(VisionEncoder, self).__init__()
        # Input: 3 x 128 x 128
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1) # 64x64
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 16x16
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 8x8
        self.bn4 = nn.BatchNorm2d(256)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 8 * 8, embed_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class NeuralNavigator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=2, max_text_len=10, output_seq_len=10):
        super(NeuralNavigator, self).__init__()
        
        self.vision_encoder = VisionEncoder(embed_dim)
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_pos_embedding = nn.Parameter(torch.randn(1, max_text_len, embed_dim))
        
        # Transformer Decoder to predict path
        # We will use the fused image+text features as memory for the decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Learnable queries for the 10 path points
        self.path_queries = nn.Parameter(torch.randn(1, output_seq_len, embed_dim))
        
        self.output_head = nn.Linear(embed_dim, 2) # Predict (x, y)

    def forward(self, image, text):
        # Image Features: (Batch, EmbedDim) -> (Batch, 1, EmbedDim)
        img_features = self.vision_encoder(image).unsqueeze(1)
        
        # Text Features: (Batch, SeqLen, EmbedDim)
        text_features = self.text_embedding(text)
        # Add positional encoding to text
        text_features = text_features + self.text_pos_embedding[:, :text_features.size(1), :]
        
        # Fuse: Concatenate Image and Text tokens -> (Batch, 1+SeqLen, EmbedDim)
        memory = torch.cat([img_features, text_features], dim=1)
        
        # Decoder Queries: (Batch, 10, EmbedDim)
        batch_size = image.size(0)
        queries = self.path_queries.expand(batch_size, -1, -1)
        
        # Transformer Decoder
        # tgt: queries, memory: fused features
        output = self.transformer_decoder(tgt=queries, memory=memory)
        
        # Predict Coordinates
        coords = self.output_head(output) # (Batch, 10, 2)
        
        return coords
