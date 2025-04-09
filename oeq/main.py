import os
import random
import shutil
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

def prepare_dataset(data_dir='Oxford-IIIT-Pet', test_size=0.1, val_size=0.1):
    print("Preparing dataset structure...")

    # Create directories if they don't exist
    os.makedirs(os.path.join(data_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'train', 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test', 'annotations'), exist_ok=True)

    # Get all image files from the original location
    original_img_dir = 'images'
    original_ann_dir = os.path.join('annotations', 'trimaps')

    if not os.path.exists(original_img_dir) or not os.path.exists(original_ann_dir):
        raise FileNotFoundError(
            "Original dataset files not found. Please download and extract images.tar.gz and annotations.tar.gz first."
        )

    # Get all jpg filenames in the directory
    image_files = []
    for f in os.listdir(original_img_dir):
        if f.endswith('.jpg'):
            image_files.append(f)

    # Shuffle the dataset
    random.shuffle(image_files)

    # Calculate portion cutoffs
    total = len(image_files)
    test_split = int(total * test_size)
    val_split = int(total * val_size)

    # Split files based on the above values
    test_files = image_files[:test_split]
    val_files = image_files[test_split:test_split + val_split]
    train_files = image_files[test_split + val_split:]

    # Helper function to copy files
    def copy_files(files, split):
        for f in files:
            # Copy image
            shutil.copy(
                os.path.join(original_img_dir, f),
                os.path.join(data_dir, split, 'images', f)
            )
            # Copy annotation (replace .jpg with .png)
            ann_file = f.replace('.jpg', '.png')
            shutil.copy(
                os.path.join(original_ann_dir, ann_file),
                os.path.join(data_dir, split, 'annotations', ann_file)
            )

    # Copy files to respective directories
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"Dataset prepared with {len(train_files)} training, {len(val_files)} validation, and {len(test_files)} test images")

def check_dataset_exists(data_dir='Oxford-IIIT-Pet'):
    required_folders = [
        os.path.join(data_dir, 'train', 'images'),
        os.path.join(data_dir, 'train', 'annotations'),
        os.path.join(data_dir, 'val', 'images'),
        os.path.join(data_dir, 'val', 'annotations')
    ]

    for folder in required_folders:
        if not os.path.exists(folder):
            return False
        if len(os.listdir(folder)) == 0:
            return False
    return True

class AnimalSegmentationDataset(Dataset):
    def __init__(self, images_dir, annotation_dir):
        self.images_dir = images_dir
        self.annotation_dir = annotation_dir

        # Image transforms (with augmentation for training)
        self.image_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Mask transforms (must preserve integer class values)
        self.mask_transform = T.Compose([
            T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
            T.PILToTensor()
        ])

        self.image_fnames = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith('.jpg')])
        self.annotation_fnames = sorted([f for f in os.listdir(self.annotation_dir) if f.lower().endswith('.png')])

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        img_name = self.image_fnames[idx]
        mask_name = img_name.replace('.jpg', '.png')

        img_path = os.path.join(self.images_dir, img_name)
        annotation_path = os.path.join(self.annotation_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        annotation = Image.open(annotation_path)

        # Apply transforms
        image = self.image_transform(image)
        annotation = self.mask_transform(annotation)
        
        # Process mask: convert to long tensor and adjust class indices
        annotation = annotation.squeeze(0).long()  # Remove channel dim and ensure correct type
        annotation -= 1  # Convert from 1-3 to 0-2 for CrossEntropyLoss

        return image, annotation

class SimpleSegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):  # Changed to 3 output channels
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        
        # Decoder
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = self.out_conv(x)
        return x

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Calculate accuracy
            preds = outputs.argmax(dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            # Calculate IoU per class
            for class_idx in range(3):  # For 3 classes
                pred_mask = (preds == class_idx)
                true_mask = (masks == class_idx)
                intersection = (pred_mask & true_mask).sum().float()
                union = (pred_mask | true_mask).sum().float()
                iou += (intersection / (union + 1e-10)).item()

    metrics = {
        'loss': running_loss / len(loader),
        'accuracy': correct_pixels / total_pixels,
        'iou': iou / (len(loader) * 3)  # Average IoU across classes and batches
    }
    return metrics

if __name__ == "__main__":
    print("Supervised Learning Code Started")

    if not check_dataset_exists('Oxford-IIIT-Pet'):
        os.makedirs('Oxford-IIIT-Pet', exist_ok=True)
        prepare_dataset(data_dir='Oxford-IIIT-Pet', test_size=0.1, val_size=0.1)

    print("Initialising Dataset Objects")
    base_dir = "Oxford-IIIT-Pet"
    train_dataset = AnimalSegmentationDataset(
        os.path.join(base_dir, "train", "images"),
        os.path.join(base_dir, "train", "annotations")
    )
    val_dataset = AnimalSegmentationDataset(
        os.path.join(base_dir, "val", "images"),
        os.path.join(base_dir, "val", "annotations")
    )
    test_dataset = AnimalSegmentationDataset(
        os.path.join(base_dir, "test", "images"),
        os.path.join(base_dir, "test", "annotations")
    )

    print("Creating Dataloaders")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    print("Setting Hyperparameters")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSegNet(in_channels=3, out_channels=3).to(device)
    lr = 0.001
    epochs = 3
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    print("Training Model")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']*100:.2f}% | Val IoU: {val_metrics['iou']:.4f}")

    # Final evaluation on test set
    print("\nTesting Model")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Pixel Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Mean IoU: {test_metrics['iou']:.4f}")