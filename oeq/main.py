import os
import random
import shutil
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

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
        raise FileNotFoundError("Original dataset files not found. Please download and extract images.tar.gz and annotations.tar.gz first")

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

    print(str(len(train_files)) + " training images, " + str(len(val_files)) + " validation images, and " + str(len(test_files)) + "test images")


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

        # Process mask: convert to binary (0=background, 1=foreground+boundary)
        annotation = annotation.squeeze(0).long()  # Remove channel dim and ensure correct type
        annotation = (annotation > 1).long()  # Convert to binary: 0=background, 1=foreground+boundary

        return image, annotation


class SupervisedNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout_prob=0.3):
        super().__init__()

        # Encoder (Downsampling)
        self.enc_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(16)  # BN after conv
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(32)  # BN after conv
        self.enc_dropout = nn.Dropout2d(p=dropout_prob)  # Spatial dropout

        # Decoder (Upsampling)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(16)  # BN after conv
        self.dec_dropout = nn.Dropout2d(p=dropout_prob / 2)  # Reduced dropout

        # Final prediction (single channel for binary output)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))  # Conv → BN → ReLU
        x = self.pool1(x)

        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_dropout(x)  # Apply dropout after last encoder layer

        # Decoder
        x = self.upsample(x)
        x = F.relu(self.dec_bn3(self.dec_conv3(x)))
        x = self.dec_dropout(x)  # Apply dropout before final conv

        # Final prediction (sigmoid activation for binary output)
        x = self.out_conv(x)
        return torch.sigmoid(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.float().to(device)  # Convert to float for BCE loss

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1))  # Add channel dimension
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            running_loss += loss.item()

            # Calculate accuracy
            preds = (outputs > 0.5).float()  # Threshold at 0.5
            correct_pixels += (preds == masks.unsqueeze(1)).sum().item()
            total_pixels += masks.numel()

    metrics = {
        'loss': running_loss / len(loader),
        'accuracy': correct_pixels / total_pixels
    }
    return metrics


def plot_and_save_history(training_loss_history, validation_loss_history, validation_accuracy_history, test_accuracy, test_loss):
    # Plot the training and validation loss
    plt.figure()
    plt.plot(training_loss_history, label='Training Loss')
    plt.plot(validation_loss_history, label='Validation Loss')
    plt.plot([], [], ' ', label="Testing Loss = " + str(test_loss))
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

    # Plot the validation accuracy over time
    plt.figure()
    plt.plot(validation_accuracy_history, label='Val Accuracy')
    plt.plot([], [], ' ', label="Test Accuracy = " + str(test_accuracy))
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()


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
    model = SupervisedNetwork(in_channels=3, out_channels=1).to(device)  # Single output channel
    lr = 0.001
    epochs = 2 # SET TO 15
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    training_loss_history = []
    validation_loss_history = []
    validation_accuracy_history = []
    print("Training Model")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_metrics['loss'])

        training_loss_history.append(train_loss)
        validation_loss_history.append(val_metrics['loss'])
        validation_accuracy_history.append(val_metrics['accuracy'])

        print("Epoch " + str(epoch + 1) + "/" + str(epochs))
        print("Train Loss: " + str(round(train_loss, 4)) + " | Val Loss: " + str(round(val_metrics['loss'], 4)))
        print("Val Accuracy: " + str(round(val_metrics['accuracy'] * 100, 2)) + "%")

    # Final evaluation on test set
    print("\nTesting Model")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\nFinal Test Results:")
    print("Loss: " + str(round(test_metrics['loss'], 4)))
    print("Test Accuracy: " + str(round(test_metrics['accuracy'] * 100, 2)) + "%")

    plot_and_save_history(training_loss_history, validation_loss_history, validation_accuracy_history, 
                         round(test_metrics['accuracy'] * 100, 2), round(test_metrics['loss'], 4))