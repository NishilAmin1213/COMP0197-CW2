import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ====================== Dataset ======================
class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, mode='train', supervised=True):
        self.root = root_dir
        self.mode = mode
        self.supervised = supervised
        self._init_transforms()
        
        self.images = []
        self.masks = []
        img_dir = os.path.join(root_dir, mode, 'images')
        ann_dir = os.path.join(root_dir, mode, 'annotations')
        
        for img_name in os.listdir(img_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(ann_dir, img_name.replace('.jpg', '.png'))
                if os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.masks.append(mask_path)

    def _init_transforms(self):
        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            lambda x: (x == 1).float()
        ])

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = self.img_transform(img)
        
        if self.supervised:
            mask = Image.open(self.masks[idx])
            mask = self.mask_transform(mask)
            return img, mask
        return img

    def __len__(self):
        return len(self.images)

# ====================== Model ======================
class HybridPetModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1, backbone.layer2,
            backbone.layer3, backbone.layer4
        )
        
        # Segmentation head with proper upsampling
        self.sup_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 1, 1)
        )
        
        # Unsupervised head
        self.unsup_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 128)
        )

    def forward(self, x, mode='both'):
        features = self.encoder(x)
        
        if mode == 'sup':
            return self.sup_head(features)
        elif mode == 'unsup':
            return self.unsup_head(features)
        return self.sup_head(features), self.unsup_head(features)

# ====================== Loss ======================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, features):
        features = F.normalize(features, dim=1)
        logits = torch.mm(features, features.t()) / self.temp
        targets = torch.arange(len(features)).to(features.device)
        return F.cross_entropy(logits, targets)

# ====================== Training ======================
def train_hybrid():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridPetModel().to(device)
    
    # Phase 1: Supervised Training
    print("=== PHASE 1: SUPERVISED TRAINING ===")
    sup_dataset = OxfordPetDataset('Oxford-IIIT-Pet', 'train', supervised=True)
    sup_loader = DataLoader(sup_dataset, batch_size=16, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    seg_criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(1):
        model.train()
        for img, mask in sup_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img, 'sup')
            
            # Ensure mask matches pred size (224x224)
            if mask.size()[-2:] != pred.size()[-2:]:
                mask = F.interpolate(mask, size=pred.size()[-2:], mode='nearest')
            
            loss = seg_criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")
    
    # Phase 2: Unsupervised Refinement
    print("\n=== PHASE 2: UNSUPERVISED REFINEMENT ===")
    unsup_dataset = OxfordPetDataset('Oxford-IIIT-Pet', 'test', supervised=False)
    unsup_loader = DataLoader(unsup_dataset, batch_size=32, shuffle=True)
    
    # Freeze supervised components
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.sup_head.parameters():
        param.requires_grad = False
        
    ssl_criterion = NTXentLoss()
    optimizer = torch.optim.Adam(model.unsup_head.parameters(), lr=1e-3)
    
    for epoch in range(20):
        model.train()
        for img in unsup_loader:
            img = img.to(device)
            features = model(img, 'unsup')
            loss = ssl_criterion(features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}: SSL Loss={loss.item():.4f}")

if __name__ == '__main__':
    train_hybrid()