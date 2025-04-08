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

    print("Dataset prepared with " + str(len(train_files))  + " training, " + str(len(val_files)) +  " validation, and " + str(len(test_files)) + " test images")


def check_dataset_exists(data_dir='Oxford-IIIT-Pet'):
    """
    Check if dataset is properly set up by looking for non-empty train/val subfolders.
    """
    required_folders = [
        os.path.join(data_dir, 'train', 'images'),
        os.path.join(data_dir, 'train', 'annotations'),
        os.path.join(data_dir, 'val', 'images'),
        os.path.join(data_dir, 'val', 'annotations')
    ]

    for folder in required_folders:
        if not os.path.exists(folder):
            return False

        # Check if folders contain at least one file
        if len(os.listdir(folder)) == 0:
            return False

    return True


class AnimalSegmentationDataset(Dataset):
    def __init__(self, images_dir, annotation_dir):
        self.images_dir = images_dir
        self.annotation_dir = annotation_dir

        # Build your default transformations here:
        self.transform = T.Compose([
            T.Resize((256, 256)),  # Example resize
            T.ToTensor(),  # Convert PIL Image to Tensor
        ])

        # Assumes that for every image, the corresponding mask has the same filename (but different extension).
        self.image_fnames = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith('.jpg')])
        self.annotation_fnames = sorted([f for f in os.listdir(self.annotation_dir) if f.lower().endswith('.png')])

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        # Get file names
        img_name = self.image_fnames[idx]
        mask_name = self.annotation_fnames[idx]

        # Load image and mask
        img_path = os.path.join(self.images_dir, img_name)
        annotation_path = os.path.join(self.annotation_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        annotation = Image.open(annotation_path)

        # Apply default transforms
        image = self.transform(image)
        annotation = self.transform(annotation)

        return image, annotation


class SimpleSegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()

        # Down/Encoder path
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # halves H and W
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        # Up/Decoder path
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))

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


if __name__ == "__main__":
    print("Supervised Learning Code Started")

    # If the Oxford-IIIT-Pet doesn't exist, create it
    if not check_dataset_exists('Oxford-IIIT-Pet'):
        os.makedirs('Oxford-IIIT-Pet', exist_ok=True)

    print("Preparing the Dataset")
    # Prepare the datasets using the prepare_dataset function above
    prepare_dataset(data_dir='Oxford-IIIT-Pet', test_size=0.1, val_size=0.1)

    # Initialize datasets
    print("Initialising Dataset Objects")
    base_dir = "Oxford-IIIT-Pet"
    train_dataset = AnimalSegmentationDataset(base_dir + "/train/images",base_dir + "/train/annotations")
    val_dataset = AnimalSegmentationDataset(base_dir + "/val/images",base_dir + "/val/annotations")
    test_dataset = AnimalSegmentationDataset(base_dir + "/test/images",base_dir + "/test/annotations")

    # Create DataLoaders
    print("Creating Dataloaders")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    print("Setting Hyperparameters")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSegNet(in_channels=3, out_channels=2).to(device)
    lr = 0.001
    epochs = 2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Training Model")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f}")

        # Optional: Validate
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"           Validation Loss: {val_loss:.4f}")

