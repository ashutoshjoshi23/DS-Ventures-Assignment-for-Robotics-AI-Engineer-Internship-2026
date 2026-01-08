# Neural Navigator

This repository contains the solution for the "Neural Navigator" assignment. The goal is to build a neural network that predicts a path (sequence of coordinates) given a 2D map image and a text command. DS Ventures Assignment: Neural Navigator — Please check the Drive link. All datasets and training images are available there: **https://drive.google.com/drive/folders/1eQeoRMXwgJ2R7wJkxhvUtCE9yu60WzE8?usp=sharing**

## Project Structure

- `data_loader.py`: Custom PyTorch DataLoader to handle images and text commands.
- `model.py`: Multi-Modal Transformer architecture (Vision Encoder + Text Encoder + Transformer Decoder).
- `train.py`: Training loop with validation split and loss logging.
- `predict.py`: Inference script to generate and visualize predictions on test images.
- `requirements.txt`: List of dependencies.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train.py
   ```
   This will save the trained model to `neural_navigator.pth` and the vocabulary to `vocab.json`. It will also generate a loss curve `training_loss.png`.

3. Run prediction:
   ```bash
   python predict.py
   ```
   This will generate prediction images in the `predictions/` directory.
<img width="1000" height="500" alt="training_loss" src="https://github.com/user-attachments/assets/39b58926-671c-401f-b3e0-a2aa417e8644" />
<img width="128" height="128" alt="pred_000086" src="https://github.com/user-attachments/assets/4389b3f8-7f77-41ee-a42d-19d13bff55c7" />
<img width="128" height="128" alt="pred_000064" src="https://github.com/user-attachments/assets/d72b46f4-eb1f-4b1e-8cb9-f1c302319896" />
<img width="128" height="128" alt="pred_000024" src="https://github.com/user-attachments/assets/0e7f9527-dbbe-4fc1-bb16-1d416bf909ea" />
<img width="128" height="128" alt="pred_000016" src="https://github.com/user-attachments/assets/1466f846-99c7-4466-9b77-b3fa45cb0fd3" />
<img width="128" height="128" alt="pred_000002" src="https://github.com/user-attachments/assets/a5320188-7fc6-41fb-bdd3-6cc4755fb589" />
<img width="391" height="66" alt="Screenshot 2026-01-08 120020" src="https://github.com/user-attachments/assets/31d9f497-0b4f-4fdd-a23d-e7c5fe313618" />
<img width="478" height="181" alt="Screenshot 2026-01-08 115934" src="https://github.com/user-attachments/assets/ec16182e-f8a7-4e2d-9c40-e41ebd0754fe" />



## Challenges & Solutions

### 1. Lack of Ground Truth in Test Data
**Challenge:** The provided `test_data` annotations did not contain the ground truth `path` or `target` coordinates, making it impossible to calculate quantitative accuracy metrics (like MSE) on the test set.
**Solution:** I implemented a random split on the training data (90% Train, 10% Validation). I used the Validation set to monitor the model's performance and report the "Accuracy" (Validation MSE Loss). This ensures we have a reliable metric to gauge model performance.

### 2. Variable Text Length vs Fixed Path Length
**Challenge:** The input text commands vary in length ("Go to the Red Circle" vs "Go to the Green Square"), while the output path is always a fixed sequence of 10 coordinates.
**Solution:** 
- **Text:** I used a simple tokenizer and padded all text sequences to a fixed length (`max_text_len=10`) to allow batch processing.
- **Path:** I used a Transformer Decoder with 10 learnable "query" tokens. These queries attend to the fused image-text features and generate the 10 path coordinates in parallel (or autoregressively, but parallel is used here for simplicity with standard Transformer Decoder).

### 3. Coordinate Normalization
**Challenge:** The path coordinates are in pixel values (0-128), which can lead to unstable training gradients.
**Solution:** I normalized the coordinates to the range [0, 1] by dividing by the image size (128). During inference, the predicted coordinates are scaled back by 128 to map them to the image.

## Bug Encountered & Debugging

**Bug:** Shape Mismatch during Feature Fusion.
**Description:** When trying to concatenate the image features and text features, I encountered a `RuntimeError: Sizes of tensors must match except in dimension 1`.
**Debugging Process:**
1. **Identify:** I checked the shapes of the tensors before concatenation using `print(img_features.shape, text_features.shape)`.
2. **Analysis:** The image encoder outputted a 2D tensor `(Batch, EmbedDim)`, while the text embeddings were 3D `(Batch, SeqLen, EmbedDim)`.
3. **Fix:** I unsqueezed the image features to add a sequence dimension, making it `(Batch, 1, EmbedDim)`. This allowed me to concatenate them along the sequence dimension, resulting in a fused feature tensor of shape `(Batch, 1+SeqLen, EmbedDim)`.

## Accuracy

**Validation MSE:** 0.0127
*Note: Lower MSE indicates better accuracy. An MSE of 0.0127 corresponds to an average error of roughly 4.5 pixels per coordinate.*
