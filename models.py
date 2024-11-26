import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
import cv2
import numpy as np
from tqdm.notebook import tqdm
from PIL import Image

class CardDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")  # Load as PIL image
        if self.transform:
            img = self.transform(img)
        return img
    

class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim=128):
        super(SimCLR, self).__init__()
        # Use layers up to layer3
        self.encoder = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
        )
        self.projection = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, out_dim, kernel_size=1)
        )

    def forward(self, x):
        features = self.encoder(x)  # Shape: (B, 256, H, W)
        projections = self.projection(features)  # Shape: (B, out_dim, H, W)
        return projections
    

def train_simclr(model, dataloader, optimizer, temperature=0.5, epochs=10, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []  # List to store losses

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            batch_size = batch.size(0)

            # Generate positive pairs with data augmentation
            xi = batch  # Original batch
            xj = batch  # Ideally, apply different augmentations here

            # Forward pass
            zi = model(xi)  # Shape: (B, out_dim, H, W)
            zj = model(xj)  # Shape: (B, out_dim, H, W)

            # Flatten spatial dimensions
            zi = zi.view(batch_size, -1)
            zj = zj.view(batch_size, -1)

            # Compute similarity matrix
            z = torch.cat([zi, zj], dim=0)
            sim_matrix = torch.mm(z, z.T) / temperature

            # Create labels
            labels = torch.arange(batch_size).to(device)
            labels = torch.cat([labels, labels], dim=0)

            # Compute loss
            loss = criterion(sim_matrix, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)  # Save loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses



def segment_image(model, image_path, n_clusters=2, device='cpu'):
    model.eval()
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image. Check the file path: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (512, 512))  # Resize to model input size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])
        img_tensor = transform(img_resized).unsqueeze(0).to(device)  # Move tensor to device

        # Pass through the model's encoder
        with torch.no_grad():
            features = model.encoder(img_tensor)  # Shape: (1, C, H, W)
            features = features.squeeze(0)  # Shape: (C, H, W)
            C, H, W = features.shape
            # Flatten spatial dimensions for clustering
            features_flat = features.permute(1, 2, 0).reshape(-1, C).cpu().numpy()  # Shape: (H*W, C)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(features_flat)  # Shape: (H*W,)

        # Reshape clusters back to (H, W)
        segmented = clusters.reshape(H, W).astype(np.int32)  # Shape: (H, W)
        return segmented, img_resized  # Return both segmentation and resized image

    except Exception as e:
        print(f"Error in segment_image: {e}")
        return None, None