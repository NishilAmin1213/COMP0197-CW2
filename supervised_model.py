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

def prepare_dataset(data_dir='oxford-iiit-pet', test_size=0.1, val_size=0.1):
    print("Preparing dataset structure...")

    # Create directories if they don't exist
    os.makedirs(os.path.join(data_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'train', 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test', 'annotations'), exist_ok=True)

    # Get all image files from the original location
    original_img_dir = os.path.join(data_dir, 'images')
    original_ann_dir = os.path.join(data_dir, 'annotations', 'trimaps')


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

    print(str(len(train_files)) + " training images, " + str(len(val_files)) + " validation images, and " + str(len(test_files)) + " test images")


def check_dataset_exists(data_dir='oxford-iiit-pet'):
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

def log_metrics(filename, epoch, metrics, stage):
    line = (
        f"Epoch {epoch} | {stage} -> "
        f"Loss: {metrics['loss']:.4f}, "
        f"Acc: {metrics['accuracy']*100:.2f}%, "
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, "
        f"Dice: {metrics['dice']:.4f}, "
        f"IoU: {metrics['iou']:.4f}"
    )
    print(line)
    with open(filename, "a") as f:
        f.write(line + "\n")


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

        # Convert to binary mask (0=background, 1=foreground+boundary)
        annotation = annotation.squeeze(0).long()  # Remove channel dim, ensure it's long
        annotation = (annotation > 1).long()       # 0=background, 1=foreground

        return image, annotation


class SupervisedNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout_prob=0.3):
        super().__init__()

        # Encoder (Downsampling)
        self.enc_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_relu2 = nn.ReLU(inplace=True)
        self.enc_dropout = nn.Dropout2d(p=dropout_prob)

        # Decoder (Upsampling)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(16)
        self.dec_relu = nn.ReLU(inplace=True)
        self.dec_dropout = nn.Dropout2d(p=dropout_prob / 2)

        # Final prediction (single channel for binary output)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = self.enc_relu1(x)
        x = self.pool1(x)

        x = self.enc_conv2(x)
        x = self.enc_bn2(x)
        x = self.enc_relu2(x)
        x = self.enc_dropout(x)

        # Decoder
        x = self.upsample(x)
        x = self.dec_conv3(x)
        x = self.dec_bn3(x)
        x = self.dec_relu(x)
        x = self.dec_dropout(x)

        # Final prediction
        x = self.out_conv(x)
        return torch.sigmoid(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.float().to(device)  # Convert to float for BCE loss

        optimizer.zero_grad()
        outputs = model(images)  # shape [B, 1, H, W]
        loss = criterion(outputs, masks.unsqueeze(1))  # Must match shape [B, 1, H, W]
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on given loader and return a dictionary of metrics:
    loss, accuracy, precision, recall, dice, iou.
    """
    model.eval()
    running_loss = 0.0

    # Counters for confusion matrix
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)              # [B, 1, H, W]
            loss = criterion(outputs, masks.unsqueeze(1))
            running_loss += loss.item()

            # Threshold the outputs at 0.5
            preds = (outputs > 0.5).long()       # [B, 1, H, W]
            masks_long = masks.long().unsqueeze(1)  # [B, 1, H, W]

            # Flatten for easier confusion matrix calculations
            preds_flat = preds.view(-1)         # shape: [B * H * W]
            masks_flat = masks_long.view(-1)    # shape: [B * H * W]

            tp += torch.sum((preds_flat == 1) & (masks_flat == 1)).item()
            fp += torch.sum((preds_flat == 1) & (masks_flat == 0)).item()
            fn += torch.sum((preds_flat == 0) & (masks_flat == 1)).item()
            tn += torch.sum((preds_flat == 0) & (masks_flat == 0)).item()

    # Compute metrics
    total_loss = running_loss / len(loader)
    total_pixels = tp + fp + fn + tn

    accuracy = 0.0
    if total_pixels > 0:
        accuracy = float(tp + tn) / float(total_pixels)

    precision = 0.0
    if (tp + fp) > 0:
        precision = float(tp) / float(tp + fp)

    recall = 0.0
    if (tp + fn) > 0:
        recall = float(tp) / float(tp + fn)

    dice = 0.0
    if (2 * tp + fp + fn) > 0:
        dice = (2.0 * float(tp)) / float((2 * tp) + fp + fn)

    iou = 0.0
    if (tp + fp + fn) > 0:
        iou = float(tp) / float(tp + fp + fn)

    metrics = {
        'loss': total_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'dice': dice,
        'iou': iou
    }
    return metrics


def plot_and_save_history(training_loss_history, validation_loss_history, validation_accuracy_history, test_metrics):
    # training_loss_history, validation_loss_history: lists of floats
    # validation_accuracy_history: list of floats
    # test_metrics: dictionary with 'accuracy' and 'loss' (optionally precision, recall, etc.)

    # Plot the training and validation loss
    plt.figure()
    plt.plot(training_loss_history, label='Training Loss')
    plt.plot(validation_loss_history, label='Validation Loss')
    plt.plot([], [], ' ', label="Testing Loss = " + str(round(test_metrics['loss'], 4)))
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()

    # Plot the validation accuracy over time
    plt.figure()
    plt.plot(validation_accuracy_history, label='Val Accuracy')
    plt.plot([], [], ' ', label="Test Accuracy = " + str(round(test_metrics['accuracy'] * 100, 2)) + "%")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_plot.png")
    plt.close()


if __name__ == "__main__":
    print("Supervised Learning Code Started")

    # Check if dataset is already structured; if not, prepare it
    if not check_dataset_exists('oxford-iiit-pet'):
        os.makedirs('oxford-iiit-pet', exist_ok=True)
        prepare_dataset(data_dir='oxford-iiit-pet', test_size=0.1, val_size=0.1)

    print("Initializing Dataset Objects")
    base_dir = "oxford-iiit-pet"
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
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Initialising Hyperparameters")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupervisedNetwork(in_channels=3, out_channels=1).to(device)  # Single output channel
    lr = 0.001
    epochs = 15
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    training_loss_history = []
    validation_loss_history = []
    validation_accuracy_history = []

    print("Training Model")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']

        # Update learning rate using scheduler
        scheduler.step(val_loss)

        training_loss_history.append(train_loss)
        validation_loss_history.append(val_loss)
        validation_accuracy_history.append(val_accuracy)

        print("\nEpoch " + str(epoch + 1) + "/" + str(epochs))
        print("Train Loss: " + str(round(train_loss, 4)) + " | Val Loss: " + str(round(val_loss, 4)) + " | Val Accuracy: " + str(round(val_accuracy, 4)) + "%")

        # LOG TRAINING METRICS
        train_metrics_dict = {
            'loss': train_loss,
            'accuracy': 0,  # not tracked in this code
            'precision': 0, # not tracked in this code
            'recall': 0,    # not tracked in this code
            'dice': 0,      # not tracked in this code
            'iou': 0        # not tracked in this code
        }
        log_metrics("log.txt", epoch + 1, train_metrics_dict, "Train")
        # LOG VALIDATION METRICS
        log_metrics("log.txt", epoch + 1, val_metrics, "Validation")

    print("\n\nValidation Results")
    print("Val Loss: " + str(round(val_loss, 4)))
    print("Val Accuracy: " + str(round(val_accuracy * 100, 2)) + "%")
    print("Val Precision: " + str(round(val_metrics['precision'], 4)))
    print("Val Recall: " + str(round(val_metrics['recall'], 4)))
    print("Val Dice: " + str(round(val_metrics['dice'], 4)))
    print("Val IoU: " + str(round(val_metrics['iou'], 4)))

    # Final evaluation on test set
    print("\n\nTest Results:")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print("Test Loss: " + str(round(test_metrics['loss'], 4)))
    print("Test Accuracy: " + str(round(test_metrics['accuracy'] * 100, 2)) + "%")
    print("Test Precision: " + str(round(test_metrics['precision'], 4)))
    print("Test Recall: " + str(round(test_metrics['recall'], 4)))
    print("Test Dice: " + str(round(test_metrics['dice'], 4)))
    print("Test IoU: " + str(round(test_metrics['iou'], 4)))

    log_metrics("log.txt", epochs, test_metrics, "Test")
    print("Test metrics logged to log.txt")

    # Plot training/validation curves, including test metrics in the legend
    plot_and_save_history(training_loss_history, validation_loss_history, validation_accuracy_history, test_metrics)

    print("Saving this model to supervised_model.pt")
    torch.save(model, "./supervised_model.pt")
