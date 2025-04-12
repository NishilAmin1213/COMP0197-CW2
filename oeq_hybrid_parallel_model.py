import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

# Set random seeds for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def prepare_dataset(data_dir='Oxford-IIIT-Pet', test_size=0.1, val_size=0.1):
    """Prepare the dataset by splitting into train/val/test and copying files"""
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
    
    image_files = [f for f in os.listdir(original_img_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)  # Shuffle for random split
    
    # Calculate split indices
    total = len(image_files)
    test_split = int(total * test_size)
    val_split = int(total * val_size)
    
    # Split files
    test_files = image_files[:test_split]
    val_files = image_files[test_split:test_split+val_split]
    train_files = image_files[test_split+val_split:]

    # Helper function to copy files
    def copy_files(files, split):
        for f in files:
            # Copy image
            shutil.copy(os.path.join(original_img_dir, f), 
                       os.path.join(data_dir, split, 'images', f))
            # Copy annotation
            ann_file = f.replace('.jpg', '.png')
            shutil.copy(os.path.join(original_ann_dir, ann_file), 
                       os.path.join(data_dir, split, 'annotations', ann_file))
    
    # Copy files to respective directories
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    print(f"Dataset prepared with {len(train_files)} training, {len(val_files)} validation, and {len(test_files)} test images")

def check_dataset_exists(data_dir='Oxford-IIIT-Pet'):
    """Check if dataset is properly set up"""
    required_folders = [
        os.path.join(data_dir, 'train', 'images'),
        os.path.join(data_dir, 'train', 'annotations'),
        os.path.join(data_dir, 'val', 'images'),
        os.path.join(data_dir, 'val', 'annotations')
    ]
    
    for folder in required_folders:
        if not os.path.exists(folder):
            return False
        
        # Check if folders contain files
        if len(os.listdir(folder)) == 0:
            return False
    
    return True

class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, mode='train', weak_supervision=True):
        self.root_dir = root_dir
        self.mode = mode
        self.weak_supervision = weak_supervision
        self.target_size = (224, 224)
        
        # Get list of images and masks
        self.images = []
        self.masks = []
        self.labels = []
        
        images_dir = os.path.join(root_dir, mode, 'images')
        masks_dir = os.path.join(root_dir, mode, 'annotations')
        
        # Verify the directories exist
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
        
        for img_name in os.listdir(images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(images_dir, img_name)
                mask_path = os.path.join(masks_dir, img_name.replace('.jpg', '.png'))
                
                if os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.masks.append(mask_path)
                    self.labels.append(1)  # All images contain animals
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            mask = Image.open(self.masks[idx])
            
            # Convert mask to binary (1: animal, 0: background)
            mask = np.array(mask)
            mask = (mask == 1).astype(np.uint8)
            
            # Apply transforms
            if self.mode == 'train':
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            
            img = transform(img)
            
            # Mask transforms
            mask = transforms.functional.to_tensor(mask)
            mask = transforms.functional.resize(mask, self.target_size)
            
            if self.weak_supervision:
                # For weak supervision, return scalar label
                return img, torch.tensor([self.labels[idx]], dtype=torch.float32)
            else:
                # For full supervision, return mask
                return img, mask
            
        except Exception as e:
            print(f"Error loading {self.images[idx]}: {str(e)}")
            random_idx = np.random.randint(0, len(self)-1)
            return self[random_idx]

def create_separate_dataloaders(data_dir, batch_size=16, num_workers=4):
    """Create separate dataloaders for weak and full supervision"""
    weak_train_dataset = OxfordPetDataset(data_dir, mode='train', weak_supervision=True)
    full_train_dataset = OxfordPetDataset(data_dir, mode='train', weak_supervision=False)
    val_dataset = OxfordPetDataset(data_dir, mode='val', weak_supervision=False)
    
    print(f"Loaded {len(weak_train_dataset)} weakly supervised training samples")
    print(f"Loaded {len(full_train_dataset)} fully supervised training samples")
    print(f"Loaded {len(val_dataset)} validation samples")
    
    weak_train_loader = DataLoader(
        weak_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return weak_train_loader, full_train_loader, val_loader

class CombinedSegmentationModel(nn.Module):
    def __init__(self, num_classes=1):
        super(CombinedSegmentationModel, self).__init__()
        
        # Shared encoder with pretrained weights
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.shared_encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
        # Weakly-supervised branch (CAM)
        self.weakly_supervised = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, num_classes, kernel_size=1)
        )
        
        # Fully-supervised branch
        self.fully_supervised = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Final segmentation head
        self.combined_head = nn.Sequential(
            nn.Conv2d(2048 + num_classes, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, num_classes, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Shared encoder
        shared_features = self.shared_encoder(x)
        
        # Weakly-supervised branch
        cam = self.weakly_supervised(shared_features)
        cam_up = nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Fully-supervised branch
        seg = self.fully_supervised(shared_features)
        seg_up = nn.functional.interpolate(seg, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Combine features
        combined = torch.cat([shared_features, seg], dim=1)
        final_seg = self.combined_head(combined)
        final_seg_up = nn.functional.interpolate(final_seg, size=(224, 224), mode='bilinear', align_corners=False)
        
        return {
            'cam': cam_up,
            'seg': seg_up,
            'final': final_seg_up
        }

def train_model(model, weak_train_loader, full_train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Alternate between weak and full supervision batches
        weak_iter = iter(weak_train_loader)
        full_iter = iter(full_train_loader)
        
        for i in range(max(len(weak_train_loader), len(full_train_loader))):
            optimizer.zero_grad()
            
            # Get next weak supervision batch if available
            try:
                weak_inputs, weak_labels = next(weak_iter)
                weak_inputs = weak_inputs.to(device)
                weak_labels = weak_labels.to(device)
                
                weak_outputs = model(weak_inputs)
                weak_loss = criterion(weak_outputs['cam'].mean(dim=(2,3)).squeeze(), weak_labels.squeeze())
            except StopIteration:
                weak_loss = None
            
            # Get next full supervision batch if available
            try:
                full_inputs, full_labels = next(full_iter)
                full_inputs = full_inputs.to(device)
                full_labels = full_labels.to(device)
                
                full_outputs = model(full_inputs)
                full_loss = criterion(full_outputs['final'], full_labels)
            except StopIteration:
                full_loss = None
            
            # Calculate combined loss
            if weak_loss is not None and full_loss is not None:
                loss = (weak_loss + full_loss) / 2
            elif weak_loss is not None:
                loss = weak_loss
            else:
                loss = full_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * (weak_inputs.size(0) if weak_loss is not None else full_inputs.size(0))
        
        epoch_loss = running_loss / len(weak_train_loader.dataset)  # Using weak loader length as reference
        train_loss_history.append(epoch_loss)
        print(f'Train Loss: {epoch_loss:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs['final'], labels)
                
                running_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)
        print(f'Val Loss: {epoch_val_loss:.4f}')
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = model.state_dict()
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Plot training history
    # plt.figure()
    # plt.plot(train_loss_history, label='Train Loss')
    # plt.plot(val_loss_history, label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig('training_history.png')
    # plt.close()
    
    print(f'Best val Loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    data_dir = 'Oxford-IIIT-Pet'
    
    # Check and prepare dataset
    if not check_dataset_exists(data_dir):
        print("Dataset not properly set up. Preparing now...")
        try:
            prepare_dataset(data_dir)
            print("Dataset preparation complete!")
        except Exception as e:
            print(f"Failed to prepare dataset: {str(e)}")
            print("Please ensure you have:")
            print("1. Downloaded images.tar.gz and annotations.tar.gz")
            print("2. Extracted them to create 'images' and 'annotations' folders")
            exit(1)
    
    # Verify dataset
    if not check_dataset_exists(data_dir):
        print("Dataset still not ready after preparation. Please check manually.")
        exit(1)
    
    print("\nChecking dataset structure...")
    print(f"Train images exist: {os.path.exists(os.path.join(data_dir, 'train', 'images'))}")
    print(f"Train annotations exist: {os.path.exists(os.path.join(data_dir, 'train', 'annotations'))}")
    print(f"Val images exist: {os.path.exists(os.path.join(data_dir, 'val', 'images'))}")
    print(f"Val annotations exist: {os.path.exists(os.path.join(data_dir, 'val', 'annotations'))}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    weak_train_loader, full_train_loader, val_loader = create_separate_dataloaders(data_dir)
    
    # Initialize model
    model = CombinedSegmentationModel().to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train the model
    model = train_model(
        model,
        weak_train_loader,
        full_train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=30
    )
    
    # Save final model
    torch.save(model.state_dict(), 'final_model_oeq_hybrid.pth')
