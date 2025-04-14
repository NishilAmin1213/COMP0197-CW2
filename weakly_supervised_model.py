"""
# 1. Setting Up
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.v2 as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import OxfordIIITPet
from torchvision.ops import box_convert, box_iou

import numpy as np
import os
import random
from PIL import Image
import xml.etree.ElementTree as ET


# External Libraries
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import cv2

# Downloads the Oxford-IIIT Pet Dataset if not already present
OxfordIIITPet(root='./', download=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
2. Load the Oxford-IIIT Pet Dataset (Classification Labels)
"""

from torchvision.datasets import OxfordIIITPet

# Custom wrapper to apply transform on-the-fly
class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, target = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


def load_classifier_dataset():

    # Define transforms
  train_transform = T.Compose([
      T.Resize((256, 256)),
      T.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
      T.RandomHorizontalFlip(p=0.5),
      T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  val_transform = T.Compose([
      T.Resize((224, 224)),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  # Load dataset without transform first
  base_dataset = OxfordIIITPet(
    root="./",
    download=True,
    target_types="category",  # classification labels only
    split="trainval",
    transform=None)


  # Train/val split
  train_size = int(0.85 * len(base_dataset))
  val_size = len(base_dataset) - train_size
  train_subset, val_subset = random_split(base_dataset, [train_size, val_size])

  # Wrap subsets with respective transforms
  train_ds = TransformDataset(train_subset, train_transform)
  val_ds = TransformDataset(val_subset, val_transform)

  # Data loaders
  train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

  # num_classes = 37  # There are 37 pet breeds
  return train_loader, val_loader

"""
3. Model Initialization (Classifier)

Pre-trained ResNet50 for classification.
"""

def get_resnet50_classifier_model(num_classes=37, pretrained_weights=True):
    """
    Oxford-IIIT has 37 categories (pet breeds) for classification.
    """
    if pretrained_weights:
      model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
      model = resnet50(weights=None)

    # Replace the final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

"""
4. Visualizations

*   plot_training_curves() - to plot acc,loss,lr after training
*   visualize_test_samples() - to visualize samples from test dataset by converting trimaps -> binary: Original image | Binary GT | Predict mask
*   visualize_segmentation_val_ds() - to visualize samples from validation dataset: Image | Pseudo Mask | Predict Mask

"""

def plot_training_curves(
    num_epochs,
    train_acc_history,
    val_acc_history,
    train_loss_history,
    val_loss_history,
    lr_per_batch,
    prefix="model",
    show=True
):
    """
    Creates and saves 3 separate figures:
      1) Accuracy vs. Epoch
      2) Loss vs. Epoch
      3) LR vs. Batch

    Inputs:
      - num_epochs: int (total training epochs)
      - train_acc_history: list of length num_epochs
      - val_acc_history:   list of length num_epochs
      - train_loss_history: list of length num_epochs
      - val_loss_history:   list of length num_epochs
      - lr_per_batch: list of LR values for each training batch
      - prefix: string prefix for saving the plots
      - show: if True, calls plt.show() for each figure
    """

    # 1) Accuracy vs. Epoch
    plt.figure()
    epochs_range = range(num_epochs)
    plt.plot(epochs_range, train_acc_history, label="Train Accuracy")
    plt.plot(epochs_range, val_acc_history,   label="Val Accuracy")
    plt.legend()
    if prefix == 'classifier':
      plt.title("Classifier Accuracy vs. No. of epochs")
    else:
      plt.title("Segmentation Pixel Accuracy vs. No. of epochs")


    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"{prefix}_accuracy_vs_epochs.png")
    if show:
        plt.show()

    # 2) Loss vs. Epoch
    plt.figure()
    plt.plot(epochs_range, train_loss_history, label="Train Loss")
    plt.plot(epochs_range, val_loss_history,   label="Val Loss")
    plt.legend()

    if prefix == 'classifier':
      plt.title("Classifier Loss vs. No. of epochs")
    else:
      plt.title("Segmentation Loss vs. No. of epochs")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{prefix}_loss_vs_epochs.png")
    if show:
        plt.show()

    # 3) Learning Rate vs. Batch
    plt.figure()
    plt.plot(lr_per_batch)

    if prefix == 'classifier':
      plt.title("Classifier Learning Rate vs. Batch no.")
    else:
      plt.title("Segmentation Learning Rate vs. Batch no.")

    plt.xlabel("Batch iteration")
    plt.ylabel("Learning Rate")
    plt.savefig(f"{prefix}_lr_vs_batch.png")
    if show:
        plt.show()

def visualize_test_samples(
    model,
    device,
    dataset,
    num_samples=5,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
):
    """
    Visualize a few samples from the test dataset:
      - Original image (de-normalized)
      - Ground truth mask (binary, ignoring boundary=2)
      - Predicted mask

    Args:
        model: trained segmentation model
        device: "cpu" or "cuda"
        dataset: a Dataset that returns (image_tensor, trimap),
                 e.g. your OxfordPetsSegmentation with transforms
        num_samples: how many samples to visualize
        mean, std: for de-normalizing images if you used ImageNet stats
    """
    model.eval()

    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    for idx in indices:
        # 1) Get the sample (image, trimap)
        image_tensor, trimap_tensor = dataset[idx]
        # image_tensor shape: [C,H,W]
        # trimap_tensor shape: [H,W], in {0,1,2} after any shift logic

        # Move image to device, add batch dimension
        input_batch = image_tensor.unsqueeze(0).to(device)

        # Model forward
        with torch.no_grad():
            outputs = model(input_batch)["out"]  # [1,2,H,W] for binary
        preds = torch.argmax(outputs, dim=1)     # [1,H,W]
        pred_mask = preds.squeeze(0).cpu().numpy()  # [H,W] in {0,1}

        # 2) Convert ground-truth trimap -> binary mask (ignore boundary=2 => set to 0 or remove)
        trimap_np = trimap_tensor.numpy()  # [H,W]
        gt_mask = np.where(trimap_np == 1, 1, 0).astype(np.uint8)

        # 3) De-normalize the image for display
        denorm_image = denormalize_image(image_tensor, mean, std)
        # shape [C,H,W], in [0,1] after clamp
        denorm_image_np = denorm_image.permute(1,2,0).cpu().numpy()  # [H,W,C]

        # 4) Plot them side by side in subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Original (de-normalized)
        axes[0].imshow(denorm_image_np)
        axes[0].set_title(f"Index {idx} - Original")
        axes[0].axis("off")

        # GT Mask (binary, boundary=ignored)
        axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth Mask (binary)")
        axes[1].axis("off")

        # Predicted
        axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig("visualize_test_sample.png")
        #plt.show()

def denormalize_image(tensor, mean, std):
    """
    Invert the normalization used for ImageNet (or custom).
    tensor: [C,H,W], assumed normalized with mean/std
    mean, std: tuples of length 3
    returns a [C,H,W] in [0,1] range (clamped)
    """
    tensor = tensor.clone()
    for c in range(tensor.shape[0]):
        tensor[c] = tensor[c]* std[c] + mean[c]

    # Clamp to [0,1] to avoid any out-of-bounds
    return torch.clamp(tensor, 0, 1)

def visualize_segmentation_val_ds(
    seg_model,
    seg_val_ds,  # A Dataset that returns (image_tensor, mask_tensor)
    device,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    indices_to_show=[0,1,2]
):
    """
    - seg_model: trained segmentation model (e.g. DeepLab) in eval mode
    - seg_val_ds: dataset returning (image_tensor, mask_tensor)
    - device: "cuda" or "cpu"
    - mean, std: used for denormalizing the image
    - indices_to_show: which dataset indices to visualize
    """
    seg_model.eval()

    for idx in indices_to_show:
        # 1) Grab one sample (image & ground truth) from the dataset
        image_tensor, gt_mask_tensor = seg_val_ds[idx]
        # image_tensor: [C,H,W], already normalized
        # gt_mask_tensor: [H,W] in {0,1} (binary) or [0..N-1] (multi-class)

        # 2) Move image to device and do inference
        input_batch = image_tensor.unsqueeze(0).to(device)  # shape [1,C,H,W]

        with torch.no_grad():
            output = seg_model(input_batch)['out']  # shape [1, num_classes, H, W]

        preds = torch.argmax(output, dim=1)  # shape [1,H,W]
        pred_mask = preds.squeeze(0).cpu().numpy().astype(np.uint8)  # shape [H,W]

        # 3) De-normalize the image for display
        denorm_image = denormalize_image(image_tensor, mean, std)  # still [C,H,W]
        denorm_image_np = denorm_image.permute(1,2,0).cpu().numpy()  # [H,W,C] in [0..1]

        # Ground truth mask
        gt_mask_np = gt_mask_tensor.cpu().numpy()

        # 4) Plot them side by side in subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Original image
        axes[0].imshow(denorm_image_np)
        axes[0].set_title(f"Original Image")
        axes[0].axis("off")

        # Ground truth mask
        axes[1].imshow(gt_mask_np, cmap="gray")
        axes[1].set_title("Pseudo GT Mask (from CAM)")
        axes[1].axis("off")

        # Predicted mask
        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Seg Model Predicted Mask")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig("visualize_segmentation_val_ds.png")
        #plt.show()

"""
5. Log Metrics
"""

import time

def log_metrics(filename, epoch, train_loss, train_acc, val_loss, val_acc, epoch_time):
    """
    Appends one line of metrics to a log file.

    filename: string path to the .txt file
    epoch: current epoch number
    train_loss, train_acc: training loss and accuracy for this epoch
    val_loss, val_acc: validation loss and accuracy for this epoch
    epoch_time: how long this epoch took (in seconds, for example)
    """

    log_entry = (f"Epoch={epoch}, "
                 f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f}, "
                 f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}, "
                 f"Time={epoch_time:.2f}s")

    # Append to file
    with open(filename, "a") as f:
        f.write(log_entry + "\n")

"""
6. Training the Classifier
"""

def train_classifier_with_metrics(
    model, train_loader, val_loader, num_epochs=10,  base_lr=1e-3, max_lr=1e-2,
    weight_decay=1e-4, clip_grad_norm=5.0, save_path="classifier.pth", log_file = "classifier_metrics_log.txt"):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # total_steps = total number of batches for the entire training
    total_steps = len(train_loader) * num_epochs

        # Initialize OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps)

    train_loss_history = []
    val_loss_history   = []
    train_acc_history  = []
    val_acc_history    = []
    lr_per_batch       = []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader: # bboxes not loaded into train loader so not used in training here
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            if clip_grad_norm is not None:
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

            scheduler.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            lr_per_batch.append(optimizer.param_groups[0]["lr"])

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc  = correct / total
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        val_loss, val_acc = evaluate_classifier_with_accuracy(model, val_loader, criterion, device)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        epoch_time = time.time() - start_time

        # log metrics
        log_metrics(
            filename=log_file,
            epoch=epoch+1,
            train_loss=epoch_train_loss,
            train_acc=epoch_train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            epoch_time=epoch_time)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Classifier model saved to {save_path}")


    # Call the common plotting function
    plot_training_curves(
        num_epochs,
        train_acc_history,
        val_acc_history,
        train_loss_history,
        val_loss_history,
        lr_per_batch,
        prefix="classifier")

    return model

def evaluate_classifier_with_accuracy(model, val_loader, criterion, device):
    """Returns val_loss, val_accuracy in training loop."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    val_loss = running_loss / len(val_loader.dataset)
    val_acc  = correct / total
    return val_loss, val_acc

"""
7. Generating Raw CAMs (Grad-CAM)

A straightforward Grad-CAM approach:
"""

class GradCAM:
    """
    Simple Grad-CAM for ResNet-based networks.
    """
    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.model.eval()

        # Hook the target layer
        self.target_layer = None
        for name, module in self.model.named_children():
            if name == target_layer_name:
                self.target_layer = module
                break

        if self.target_layer is None:
            raise ValueError(f"Layer {target_layer_name} not found in model")

        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        """
        x: input image tensor of shape [B, C, H, W]
        class_idx: which class index to compute CAM for. If None, uses argmax
        Returns: CAM for each image in the batch
        """
        logits = self.model(x)  # forward pass
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)

        # Compute gradients w.r.t. target class
        one_hot = torch.zeros_like(logits)
        for i in range(logits.size(0)):
            one_hot[i, class_idx[i]] = 1.0

        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients  # [B, C, H', W']
        activations = self.activations  # [B, C, H', W']

        # Global-average-pool the gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # Weighted sum of activations
        cams = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H', W']

        # ReLU
        cams = F.relu(cams)

        # Normalize each CAM to [0,1]
        cams = cams - cams.view(cams.size(0), -1).min(dim=1)[0].view(cams.size(0),1,1,1)
        cams = cams / (cams.view(cams.size(0), -1).max(dim=1)[0].view(cams.size(0),1,1,1) + 1e-8)

        return cams

"""
Bounding Box Usage:
  *	Bounding box expansion helps ensure that most of the pet is inside the bounding box.
  *	Soft weighting ensures that if there’s any part of the pet outside that expanded box, 
  we don’t completely discard it (i.e., zero it out). Instead, we downweight it, 
  but we still keep it if the model strongly activates there.

Multi_Scale_CAM:

multi_scale_cam loops over each scale in scales, resizing the input images, 
calling gradcam(...), and then upsampling the resulting CAM back to the original shape.
We stack and average them, producing a single combined CAM for each image in the batch.
"""

import torch
import torch.nn.functional as F

def multi_scale_cam(gradcam, images, class_idx, scales=[0.75, 1.0, 1.25]):
    """
    images: [B, C, H, W] in torch format
    class_idx: [B] (which class to compute CAM for each image)
    scales: list of float scaling factors
    returns: a single combined CAM per image [B,1,H,W] in [0,1]
    """
    device = images.device
    B, C, H, W = images.shape

    all_cams = []
    for scale in scales:
        # 1) Resize images
        new_h = int(H * scale)
        new_w = int(W * scale)
        scaled_imgs = F.interpolate(images, size=(new_h, new_w),
                                    mode='bilinear', align_corners=False)

        # 2) Run GradCAM for each scaled image
        with torch.enable_grad():
            scaled_cams = gradcam(scaled_imgs, class_idx=class_idx)
            # scaled_cams shape: [B, 1, new_h', new_w']

        # 3) Upsample each CAM back to original (H,W)
        up_cams = F.interpolate(scaled_cams, size=(H, W),
                                mode='bilinear', align_corners=False)
        all_cams.append(up_cams)

    # 4) Average all scaled CAMs
    combined = torch.mean(torch.stack(all_cams, dim=0), dim=0)  # shape [B,1,H,W]
    combined = torch.clamp(combined, 0, 1)  # ensure [0,1]
    return combined

def expand_bbox(bbox, expansion_ratio=1.5, img_w=224, img_h=224):
    """
    Expand the bounding box by 'expansion_ratio' while staying within [0, img_w]x[0, img_h].
    bbox: (x_min, y_min, x_max, y_max)
    expansion_ratio: e.g., 1.5 -> 50% bigger in width/height each side.
    """
    x_min, y_min, x_max, y_max = bbox

    # Current width/height
    box_w = x_max - x_min
    box_h = y_max - y_min

    # Expansion amount
    dw = (box_w * (expansion_ratio - 1)) / 2.0
    dh = (box_h * (expansion_ratio - 1)) / 2.0

    # Center-based expansion
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    new_xmin = max(0, int(cx - (box_w/2 + dw)))
    new_xmax = min(img_w, int(cx + (box_w/2 + dw)))
    new_ymin = max(0, int(cy - (box_h/2 + dh)))
    new_ymax = min(img_h, int(cy + (box_h/2 + dh)))

    return (new_xmin, new_ymin, new_xmax, new_ymax)

def apply_expanded_bbox_soft_weights(cam, bbox, H, W,
                                     expansion_ratio=1.5,
                                     inside_weight=0.9,
                                     outside_weight=0.3):
    """
    1) Expand the bounding box by 'expansion_ratio'.
    2) Apply soft weighting inside vs. outside the expanded box.
       - inside_weight = 0.9
       - outside_weight = 0.3 
    """
    # Expand the bbox
    x_min, y_min, x_max, y_max = expand_bbox(bbox, expansion_ratio, img_w=W, img_h=H)

    # Create a weight map
    weights = np.full((H, W), outside_weight, dtype=np.float32)
    weights[y_min:y_max, x_min:x_max] = inside_weight

    return cam * weights

def generate_cams(model, data_loader, gradcam, device,
                  apply_bbox=True, output_dir="cams_out", scales=[0.75, 1.0, 1.25]):
    """
    Generate and save CAMs.
    We'll upsample the CAM to the original image size,
    then optionally mask it with the bounding box.
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    gradcam.model.eval()

    with torch.no_grad():
        for batch_idx, (images, targets, paths) in enumerate(data_loader):
            # images shape: [B, C, H, W]
            images = images.to(device)


            class_idxs = []
            for t in targets:
                # If there's exactly 1 label per image
                class_idxs.append(t["labels"][0])

            class_idxs = torch.tensor(class_idxs, dtype=torch.long, device=device)

            with torch.enable_grad():
                # cams_batch = gradcam(images, class_idx=class_idxs)
                cams_batch = multi_scale_cam(gradcam, images, class_idx=class_idxs, scales=scales)


            # using bilinear interpolation
            upsampled_cams = F.interpolate(cams_batch, size=(images.shape[2], images.shape[3]),
                                           mode='bilinear', align_corners=False)

            # Now upsampled_cams is [B, 1, H, W]
            upsampled_cams = upsampled_cams.squeeze(1).cpu().numpy()  # shape [B, H, W]

            # Loop over each image in the batch
            for b in range(len(images)):
                cam_2d = upsampled_cams[b]


                if apply_bbox:

                    bbox = targets[b]["boxes"][0]  # shape [4]
                    bbox = bbox.cpu().numpy()

                    cam_2d = apply_expanded_bbox_soft_weights(
                                cam_2d,
                                bbox,
                                H=cam_2d.shape[0],
                                W=cam_2d.shape[1],
                                expansion_ratio=1.5,
                                inside_weight=1.0,
                                outside_weight=0.6
                            )

                base_name = os.path.splitext(os.path.basename(paths[b]))[0]
                cam_file = f"{base_name}_cam.npy"
                cam_path = os.path.join(output_dir, cam_file)

                # Save each CAM
                np.save(cam_path, cam_2d)

    print("CAM generation complete!")

"""
8. Apply ReCAM (Refinement / Expansion)

A simplified version of Offcial ReCAM Technique that scales up smaller areas, and add expansions.
"""

def recam_refinement(cam, expansion_factor=1.2, threshold=0.3):
    # cam: 2D np array [H, W] in [0,1]
    # Step 1: thresholding
    mask = (cam >= threshold).astype(np.uint8)
    coverage = mask.sum() / (cam.shape[0]*cam.shape[1])

    # If coverage < some ratio, inflate the activation
    if coverage < 0.1:
        cam = cam * expansion_factor
        cam = np.clip(cam, 0, 1)

    # Re-threshold
    return cam

def refine_cams_with_recam(cam_dir, refined_dir="cams_refined",
                           threshold=0.3, expansion_factor=1.2):
    os.makedirs(refined_dir, exist_ok=True)

    cam_files = [f for f in os.listdir(cam_dir) if f.endswith('.npy')]
    for cfile in cam_files:
        cam_path = os.path.join(cam_dir, cfile)
        cam = np.load(cam_path)

        refined_cam = recam_refinement(cam, expansion_factor, threshold)

        # Save
        refined_path = os.path.join(refined_dir, cfile)  # same base name
        np.save(refined_path, refined_cam)

    print("ReCAM refinement complete!")



"""
9. Train Segmentation Model (DeepLab V3+)
"""

# load the Oxford-IIIT Pet dataset with bounding boxes and apply basic transformations using albumentations

root_dir = "./"  # or any folder you choose

dummy_dataload = torchvision.datasets.OxfordIIITPet(
    root=root_dir,
    split="trainval",
    target_types=["category"],  # or ["segmentation"] / ["category", "segmentation"]
    download=True,
)

def parse_xml_for_bbox(xml_file):
    """Parse the Oxford-IIIT Pet XML file for bounding box [xmin, ymin, xmax, ymax]."""
    if not os.path.exists(xml_file):
        return None, None  # Indicate missing
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    cls = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        bboxes.append([xmin, ymin, xmax, ymax])
        cls.append(name)

    return bboxes, cls

def read_split_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n')
    name= [line.split()[0] for line in lines if line.strip()]
    breed= [line.split()[1] for line in lines if line.strip()]
    return name, breed

def build_annotations_list(root_dir, txt_filename):

    images_dir = os.path.join(root_dir, "images")
    xmls_dir   = os.path.join(root_dir, "annotations", "xmls")
    txt_path   = os.path.join(root_dir, "annotations", txt_filename)

    image_bases, labels = read_split_file(txt_path)

    annotation_dicts = []
    for base in image_bases:
        xml_file  = os.path.join(xmls_dir, f"{base}.xml")
        image_file = os.path.join(images_dir, f"{base}.jpg")
        label = labels[image_bases.index(base)]

        bboxes, cls = parse_xml_for_bbox(xml_file)
        if bboxes is None or cls is None:
            print(f"Skipping missing file: {xml_file}")
            continue

        # Also check if image_file actually exists
        if not os.path.exists(image_file):
            print(f"Skipping missing image: {image_file}")
            continue

        annotation_dicts.append({
            'image_file': image_file,
            'boxes': bboxes,       # list of [xmin, ymin, xmax, ymax]
            'labels': label       # breeds (classes indexed - [1,1,1...,2,2,2..3..])
        })
    return annotation_dicts

def load_oxford_pet_annotations(root_dir):
    trainval_anns = build_annotations_list(root_dir, "trainval.txt")
    return trainval_anns

class OxfordPetBboxDatasetAlbumentations(torch.utils.data.Dataset):
    def __init__(self, annotations, transform=None):
        """
        Args:
            annotations: list of dicts, each like:
                {
                  'image_file': '/path/to/img.jpg',
                  'boxes': [[xmin, ymin, xmax, ymax], ...],
                  'labels': ['1', ['2']...]
                }
            transform: An Albumentations transform pipeline.
        """
        self.annotations = annotations
        self.transform = transform


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = ann['image_file']
        boxes = ann['boxes']  # List of bounding boxes in pascal_voc format
        labels = ann['labels']  
        # Load image as a NumPy array.
        image = np.array(Image.open(img_path).convert('RGB'))

        # Duplicate labels for each bounding box to match the length of bboxes
        labels = [labels] * len(boxes)  # This now creates a list of labels


        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']



        # Convert bounding boxes to a tensor.
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor([int(int(l)-1) for l in labels], dtype=torch.int64)


        return image, {"boxes": boxes_tensor, "labels": labels_tensor}, img_path

albumentations_transform = A.Compose(
    [
        A.Resize(224, 224), # Resize to 224x224
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),  # Converts the image to a PyTorch tensor
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
)


def detection_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    paths = [item[2] for item in batch]
    images = torch.stack(images, dim=0)  
    return images, targets, paths

def get_dataloader():
    trainval_anns = load_oxford_pet_annotations("oxford-iiit-pet")
    trainval_dataset = OxfordPetBboxDatasetAlbumentations(trainval_anns, transform=albumentations_transform)

    img_paths = []
    for i in range(len(trainval_anns)):
        img_paths.append(trainval_anns[i]['image_file'])

    # DataLoader
    train_loader = DataLoader(trainval_dataset, batch_size=8, shuffle=False,  num_workers=2, collate_fn=detection_collate_fn)
    return train_loader, img_paths



seg_transform = A.Compose([
    # 1) Resize - to have a consistent baseline size
    A.Resize(224, 224),

    # 2) Horizontal Flip (50% chance)
    A.HorizontalFlip(p=0.5),

    # 3) Optional color augmentations that affect ONLY the image
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

    # 4) Normalize the image (typical ImageNet stats)
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),

    # 5) Convert both image and mask to torch tensors
    ToTensorV2()
],
additional_targets={"mask": "mask"})

import os

def build_img_mask_pairs(img_dir, mask_dir, img_paths):
    pairs = []
    for ipath in img_paths:
        base_name = os.path.splitext(os.path.basename(ipath))[0]
        mask_name = f"{base_name}_cam.png"
        mask_path = os.path.join(mask_dir, mask_name)
        pairs.append((ipath, mask_path))
    return pairs

class PseudoSegDataset(torch.utils.data.Dataset):
    def __init__(self, img_mask_pairs, transform=None):
        self.img_mask_pairs = img_mask_pairs # list of (image_path, mask_path)
        self.transform = transform

    def __len__(self):
        return len(self.img_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.img_mask_pairs[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # Load pseudo mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)   # 0/255
        mask = (mask > 127).astype(np.uint8) # binarize as 0/1


        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


def get_deeplab_v3(num_classes=2, pretrained_weights=True):

    if pretrained_weights:
      model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    else:
      model = deeplabv3_resnet50(weights=None, weights_backbone=None, progress=False)
      # Using strict=False due to aux_classifier key issue

    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # output channels = 2 (background, foreground)
    return model

"""Training loop for segmentation:"""

def train_segmentation_with_metrics(
    seg_model,
    seg_train_loader,
    seg_val_loader,
    num_epochs=10,
    base_lr=1e-3,
    max_lr=1e-2,
    weight_decay=1e-4,
    save_path="seg_model.pth", log_file = "segmentation_metrics_log.txt"):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = seg_model.to(device)

    optimizer = optim.Adam(seg_model.parameters(), lr=base_lr, weight_decay=weight_decay)

    total_steps = len(seg_train_loader) * num_epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss() 

    train_loss_history = []
    val_loss_history   = []
    train_acc_history  = []
    val_acc_history    = []
    lr_per_batch       = []

    for epoch in range(num_epochs):

        start_time = time.time()

        seg_model.train()
        running_loss = 0.0
        correct_px   = 0
        total_px     = 0

        for images, masks in seg_train_loader:
            images = images.to(device)    # [B, C, H, W]
            masks  = masks.long().to(device)  # [B, H, W] in {0,1} for binary

            optimizer.zero_grad()
            outputs = seg_model(images)['out']  # [B, 2, H, W] if binary
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            scheduler.step()  


            running_loss += loss.item() * images.size(0)

            # Pixel-wise accuracy (for binary, predicted label is argmax among [0,1])
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]
            correct_px += (preds == masks).sum().item()
            total_px   += masks.numel()

            current_lr = optimizer.param_groups[0]["lr"]
            lr_per_batch.append(current_lr)

        train_loss = running_loss / len(seg_train_loader.dataset)
        train_acc  = correct_px / total_px  # pixel accuracy
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # Validation
        val_loss, val_acc = evaluate_segmentation_with_accuracy(seg_model, seg_val_loader, criterion, device)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        epoch_time = time.time() - start_time
        # log metrics
        log_metrics(
            filename=log_file,
            epoch=epoch+1,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            epoch_time=epoch_time)

        print(f"[{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save model
    torch.save(seg_model.state_dict(), save_path)
    print(f"Segmentation model saved to {save_path}")

    # Call the common plotting function
    plot_training_curves(
        num_epochs,
        train_acc_history,
        val_acc_history,
        train_loss_history,
        val_loss_history,
        lr_per_batch,
        prefix="segmentation")


    return seg_model

def evaluate_segmentation_with_accuracy(seg_model, seg_val_loader, criterion, device):
    """Compute val_loss and pixel accuracy for binary segmentation in training loop."""
    seg_model.eval()
    running_loss = 0.0
    correct_px   = 0
    total_px     = 0

    with torch.no_grad():
        for images, masks in seg_val_loader:
            images = images.to(device)
            masks  = masks.long().to(device)

            outputs = seg_model(images)['out']
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct_px += (preds == masks).sum().item()
            total_px   += masks.numel()

    val_loss = running_loss / len(seg_val_loader.dataset)
    val_acc  = correct_px / total_px
    return val_loss, val_acc

"""
 10. Evaluation of Segmentation Model - Metrics

Computing precision, recall, pixel_accuracy, dice, iou
*   evaluate_segmentation_metrics() - compute metrics on test dataset (images, trimap GT) by converting trimaps to binary (by ignoring the boundary).
*   evaluate_seg_val_loader() - compute metrics on validation set (images, binary pseudo maps)
"""

class OxfordPetsSegmentation(Dataset):
    def __init__(self, root, split='test', transform=None):

        # Load the base dataset with only segmentation masks
        self.dataset = OxfordIIITPet(
            root=root,
            download=True,
            target_types="segmentation",
            split=split
        )
        self.transform = transform

    def __getitem__(self, idx):
        # OxfordIIITPet returns: (PIL Image, PIL Mask)
        image_pil, mask_pil = self.dataset[idx]

        # Convert PIL -> NumPy
        image = np.array(image_pil)           
        mask  = np.array(mask_pil, dtype=np.uint8) 


        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # The trimap is typically 1=pet, 2=border, 3=background
        # Shift to 0-based: (0=pet, 1=border, 2=background)
        mask = mask - 1

        return image, mask

    def __len__(self):
        return len(self.dataset)

def evaluate_segmentation_metrics(model, device):
  """
  Evaluates the segmentation model on the test set where
  ground truth masks are trimaps (0=BG, 1=FG, 2=Boundary).

  Ignores boundary pixels for scoring.

  Returns a dict of {precision, recall, accuracy, dice, iou}.
  """

  seg_test_transform = A.Compose([
      # 1) Resize - to have a consistent baseline size
      A.Resize(224, 224),

      # 2) Normalize the image (typical ImageNet stats)
      A.Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),

      # 3) Convert both image and mask to torch tensors
      ToTensorV2()
  ],  additional_targets={"mask": "mask"})


  test_dataset = OxfordPetsSegmentation(
      root="./",
      split="test",
      transform=seg_test_transform,
  )
  print("Test size:", len(test_dataset))
  test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

  model.eval()

  # Running totals for confusion matrix
  total_tp = 0
  total_fp = 0
  total_tn = 0
  total_fn = 0

  with torch.no_grad():
      for images, trimaps in test_loader:
          images  = images.to(device)
          # forward pass
          outputs = model(images)["out"]  # shape [B, 2, H, W] if binary
          preds   = torch.argmax(outputs, dim=1)  # shape [B, H, W], in {0,1}
          preds = 1 - preds
          preds   = preds.cpu().numpy()
          trimaps = trimaps.numpy()

          # For each sample in the batch
          for b in range(images.size(0)):
              pred_mask = preds[b]       # shape [H, W], {0,1}
              gt_trimap = trimaps[b]     # shape [H, W], {0,1,2}

              # Create a mask ignoring boundary pixels (where gt_trimap==2)
              ignore_mask = (gt_trimap == 2)

              # Flatten
              pred_flat = pred_mask.flatten()        # {0,1}
              gt_flat   = gt_trimap.flatten()        # {0,1,2}
              ignore_flat = ignore_mask.flatten()    # bool

              # Filter out boundary indices
              valid_idx = np.where(~ignore_flat)[0]  # indices not boundary
              if valid_idx.size == 0:
                  # entire image might be boundary or something unusual
                  continue

              pred_valid = pred_flat[valid_idx]  # {0,1}
              gt_valid   = gt_flat[valid_idx]    # {0,1}

              # Now compute confusion matrix for these pixels
              tp = np.sum((pred_valid == 1) & (gt_valid == 1))
              fp = np.sum((pred_valid == 1) & (gt_valid == 0))
              tn = np.sum((pred_valid == 0) & (gt_valid == 0))
              fn = np.sum((pred_valid == 0) & (gt_valid == 1))

              total_tp += tp
              total_fp += fp
              total_tn += tn
              total_fn += fn

  # Compute metrics
  eps = 1e-7  # for safe division
  precision = total_tp / (total_tp + total_fp + eps)
  recall    = total_tp / (total_tp + total_fn + eps)  
  accuracy  = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)
  dice      = 2.0 * total_tp / (2.0 * total_tp + total_fp + total_fn + eps)
  iou_foreground  = total_tp / (total_tp + total_fp + total_fn + eps)
  iou_background  = total_tn / (total_tn + total_fp + total_fn + eps)
  mean_iou        = (iou_foreground + iou_background) / 2.0


  metrics_dict_test = {
      "precision": precision,
      "recall": recall,
      "accuracy": accuracy,
      "dice": dice,
      "iou_foreground": iou_foreground,
      "iou_background": iou_background,
      "mean_iou": mean_iou
  }


  # Now visualize a few random samples
  visualize_test_samples(
        model=model,
        device=device,
        dataset=test_dataset,  # same dataset used in the test_loader
        num_samples=5
    )

  return metrics_dict_test

import torch
import numpy as np

def evaluate_seg_val_loader(
    model,
    val_loader,   # yields (images, binary_masks)
    device="cuda"
):
    """
    Evaluates a binary segmentation model on a val_loader with pseudo binary GT masks in {0,1}.

    Computes the following metrics:
      1) Precision
      2) Recall (Sensitivity)
      3) Accuracy
      4) Dice
      5) IoU
    """
    model.eval()

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    with torch.no_grad():
        for images, gt_masks in val_loader:
            images = images.to(device)
            # forward
            outputs = model(images)['out']  # shape [B, 2, H, W] if binary
            preds = torch.argmax(outputs, dim=1)  # [B, H, W], in {0,1}

            preds = preds.cpu().numpy()      # shape [B,H,W]
            gts   = gt_masks.numpy()         # shape [B,H,W], in {0,1}

            # For each sample in the batch
            for b in range(images.size(0)):
                pred_mask = preds[b].flatten()  # [H*W]
                gt_mask   = gts[b].flatten()    # [H*W]

                tp = np.sum((pred_mask == 1) & (gt_mask == 1))
                fp = np.sum((pred_mask == 1) & (gt_mask == 0))
                tn = np.sum((pred_mask == 0) & (gt_mask == 0))
                fn = np.sum((pred_mask == 0) & (gt_mask == 1))

                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn

    # compute metrics
    eps = 1e-7
    precision       = total_tp / (total_tp + total_fp + eps)
    recall          = total_tp / (total_tp + total_fn + eps)
    accuracy        = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + eps)
    dice            = 2.0 * total_tp / (2.0 * total_tp + total_fp + total_fn + eps)
    iou_foreground  = total_tp / (total_tp + total_fp + total_fn + eps)
    iou_background  = total_tn / (total_tn + total_fp + total_fn + eps)
    mean_iou        = (iou_foreground + iou_background) / 2.0


    metrics_dict = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "dice": dice,
        "iou_foreground": iou_foreground,
        "iou_background": iou_background,
        "mean_iou": mean_iou
    }

    print("Validation (Binary) Metrics:")
    for k,v in metrics_dict.items():
        print(f"  {k}: {v:.4f}")

    return metrics_dict

def evaluate_on_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = get_deeplab_v3(num_classes=2, pretrained_weights=False)
    seg_model.load_state_dict(torch.load("weakly_segmentation.pth", map_location=device), strict=False)
    seg_model = seg_model.to(device)
    metrics = evaluate_segmentation_metrics(seg_model, device)

    print("Evaluation on Test Set (Ignoring Boundary Pixels):")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Dice:      {metrics['dice']:.4f}")
    print(f"  IoU_foreground:       {metrics['iou_foreground']:.4f}")
    print(f"  IoU_background:       {metrics['iou_background']:.4f}")
    print(f"  mIoU:             {metrics['mean_iou']:.4f}")


"""
11. DINO Pseudo Mask Generation
Using DINO self-attention to generate pseudo masks.
"""
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
dino_model.eval()

dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


def get_dino_attention_map(model, input_tensor):
    """
    Extracts the self-attention map from the last transformer block of a DINO ViT.
    """
    attn = None
    def hook_fn(module, input, output):
        nonlocal attn
        attn = output[1]

    handle = model.blocks[-1].attn.register_forward_hook(hook_fn)
    _ = model(input_tensor)
    handle.remove()

    if attn is None:
        raise RuntimeError("Attention map was not captured.")

    # Average across heads
    attn = attn.mean(dim=1)  # [B, tokens, tokens]
    cls_attn = attn[:, 0, 1:]  # from class token to patch tokens
    num_patches = cls_attn.shape[-1]
    h = w = int(np.sqrt(num_patches))
    cls_attn = cls_attn.reshape(-1, h, w)

    # Normalize to [0,1]
    min_val = cls_attn.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_val = cls_attn.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    cls_attn = (cls_attn - min_val) / (max_val - min_val + 1e-8)

    return cls_attn

def process_image_to_pseudo_mask(image_path, threshold=0.5):
    """
    Process an input image to get a pseudo mask using DINO self-attention.
    Returns the patch-space mask [1,h,w].
    """

    image = Image.open(image_path).convert("RGB")
    input_tensor = dino_transform(image).unsqueeze(0)  # [1,3,224,224]

    with torch.no_grad():
        attn_map = get_dino_attention_map(dino_model, input_tensor)

    # Instead of binarizing at patch-space, we keep continuous values for smoother upscaling
    continuous_mask = attn_map

    return continuous_mask  # shape [1,h,w] in patch space

def generate_dino_pseudo_masks(
    image_paths,
    output_dir="dino_pseudo_masks",
    threshold=0.3,
    upscale=True,
    final_size=(224,224)
):
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        # e.g. /path/to/image.jpg
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_file = os.path.join(output_dir, f"{base_name}_dino.png")

        continuous_mask = process_image_to_pseudo_mask(img_path, threshold=threshold)
        # Convert to numpy array
        mask_np = continuous_mask.squeeze(0).cpu().numpy()  # shape [h, w] with continuous values in [0,1]

        if upscale:
            mask_np = cv2.resize(
                mask_np,
                final_size,
                interpolation=cv2.INTER_LINEAR
            )
        # Apply Gaussian Blur 
        kernel_size = (5, 5)  
        smoothed_mask = cv2.GaussianBlur(mask_np, kernel_size, 0)

        # Convert continuous mask to binary using a threshold.
        binary_mask = (smoothed_mask * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(binary_mask, int(threshold * 255), 255, cv2.THRESH_BINARY)

        # Apply morphological opening to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Apply morphological closing to fill small holes
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

        # Save the processed mask
        cv2.imwrite(mask_file, mask_closed)

    print("DINO pseudo-masks saved in", output_dir)

"""
12. Combine DINO and ReCAM maps
This function combines DINO and ReCAM maps to produce a single pseudo mask.
"""

def combine_dino_recam(
    dino_dir,         # e.g. "dino_pseudo_masks"
    recam_dir,        # e.g. "recam_maps"
    output_dir="combined_masks",
    combine_mode="average",  # or "max"
    alpha=0.5,
    threshold=0.5
):
    """
    Combine DINO and ReCAM maps per image to produce a single pseudo mask.
    Args:
        dino_dir (str): Directory where the DINO masks are saved.
        recam_dir (str): Directory where the ReCAM masks are saved.
        output_dir (str): Where to save the combined masks.
        combine_mode (str): "average" or "max".
        alpha (float): weighting for the average, e.g. 0.5 means equal weights.
        threshold (float): final threshold for the combined map in [0..1].
    """
    os.makedirs(output_dir, exist_ok=True)

    # List all files in dino_dir. We assume matching filenames exist in recam_dir
    dino_files = [f for f in os.listdir(dino_dir) if f.endswith(".png") or f.endswith(".jpg")]

    print(len(dino_files))

    for f in dino_files:
        dino_path = os.path.join(dino_dir, f)

        recam_path = os.path.join(recam_dir, f.replace('_dino.png', '_cam.npy'))


        if not os.path.exists(recam_path):
            # skip if recam file doesn't exist
            print("skipping as recam file doesn't exist")
            continue

        # Load the DINO mask
        dino_mask = cv2.imread(dino_path, cv2.IMREAD_GRAYSCALE)  # shape [H,W], 0..255
        if dino_mask is None:
            print(f"Cannot read DINO mask: {dino_path}")
            continue

        # Load the ReCAM mask
        # recam_mask = cv2.imread(recam_path, cv2.IMREAD_GRAYSCALE)  # shape [H,W], 0..255
        recam_float = np.load(recam_path)  # shape [H,W], 0..1
        if recam_float is None:
            print(f"Cannot read ReCAM mask: {recam_path}")
            continue

        # Convert to float [0..1] if they're [0..255].
        dino_float = dino_mask.astype(np.float32) / 255.0
        # recam_float = recam_mask.astype(np.float32) / 255.0

        # Combine
        if combine_mode == "average":
            combined = alpha * dino_float + (1.0 - alpha) * recam_float
        elif combine_mode == "max":
            combined = np.maximum(dino_float, recam_float)
        else:
            raise ValueError("combine_mode must be 'average' or 'max'.")

        # Threshold the final map
        final_bin = (combined >= threshold).astype(np.uint8) * 255

        # Apply morphological opening to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_opened = cv2.morphologyEx(final_bin, cv2.MORPH_OPEN, kernel)

        # Apply morphological closing to fill small holes
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

        base_name_recam = os.path.splitext(os.path.basename(recam_path))[0]

        mask_file = os.path.join(output_dir, f"{base_name_recam}.png")

        cv2.imwrite(mask_file, mask_closed)

    print(f"Combined masks saved in: {output_dir}")


"""
Main
"""

def main():
    # 1. Load your train/val data for classifier
    train_loader, val_loader = load_classifier_dataset()

    # 2. Create a classifier and train it
    classifier = get_resnet50_classifier_model(num_classes=37,pretrained_weights=True)
    classifier = train_classifier_with_metrics(
       model=classifier,
       train_loader=train_loader,
       val_loader=val_loader,
       num_epochs=20, base_lr=1e-3, max_lr=1e-2, weight_decay=1e-4, clip_grad_norm=None, save_path="classifier.pth", log_file = "classifier_metrics_log.txt")

    # Loading trained classifier (weights file requried)
    # classifier = get_resnet50_classifier_model(num_classes=37,pretrained_weights=False)
    # classifier.load_state_dict(torch.load("classifier.pth", map_location=device))
    # classifier = classifier.to(device)


    # Load trainval dataset with Bounding Boxes
    train_loader_bbox, train_img_paths = get_dataloader()

    # 3. Generate raw CAMs
    gradcam = GradCAM(classifier, target_layer_name="layer4")  # for ResNet
    generate_cams(
        model=classifier,
        data_loader=train_loader_bbox,   # or a combined trainval loader if you want CAM for all
        gradcam=gradcam,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        apply_bbox=True,
        output_dir="cams_out", scales=[0.75, 1.0, 1.25]
    )

    # # 4. Apply ReCAM refinement
    refine_cams_with_recam("cams_out", refined_dir="cams_refined",
                           threshold=0.2, expansion_factor=1.2)


    # # 5. Generate DINO pseudo masks with smoothed edges
    generate_dino_pseudo_masks(
        image_paths=train_img_paths,
        output_dir="dino_pseudo_masks",
        threshold=0.3,
        upscale=True,
        final_size=(224,224)  # or the original size if you prefer
    )

    # 6. Combine DINO pseudo masks with CAM's
    combine_dino_recam(
        dino_dir="dino_pseudo_masks",
        recam_dir="cams_refined",
        output_dir="combined_pseudo_masks",
        combine_mode="average",
        alpha=0.5,    # only used if combine_mode="average"
        threshold=0.5
    )

    # 7. Train a segmentation model (DeepLab) with these pseudo masks

    # Build a dataset from your images and "pseudo_masks"

    img_mask_pairs = build_img_mask_pairs(
    img_dir="oxford-iiit-pet/images",
    mask_dir="combined_pseudo_masks",
    img_paths=train_img_paths) # the same list used in the dataset

    torch.cuda.empty_cache()

    seg_trainval_dataset = PseudoSegDataset(img_mask_pairs, transform=seg_transform)

    # Split dataset into train and val
    seg_train_size = int(0.85 * len(seg_trainval_dataset))
    seg_val_size = len(seg_trainval_dataset) - seg_train_size
    seg_train_ds, seg_val_ds = random_split(seg_trainval_dataset, [seg_train_size, seg_val_size])

    seg_train_loader = DataLoader(seg_train_ds, batch_size=32, shuffle=True, num_workers=2)
    seg_val_loader = DataLoader(seg_val_ds, batch_size=32, shuffle=True, num_workers=2)


    seg_model = get_deeplab_v3(num_classes=2, pretrained_weights=True)
    seg_model = train_segmentation_with_metrics(
        seg_model, seg_train_loader, seg_val_loader,
        num_epochs=20, base_lr=1e-3, max_lr=1e-2, weight_decay=1e-4, save_path="weakly_segmentation.pth", log_file = "segmentation_metrics_log.txt")

    # Loading trained Segmentation model (weights file requried)
    # seg_model = get_deeplab_v3(num_classes=2,pretrained_weights=False)
    # state_dict = torch.load("weakly_segmentation.pth", map_location=device)
    # # Remove keys related to the aux classifier
    # state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
    # seg_model.load_state_dict(state_dict)
    # seg_model = seg_model.to(device)

    # 7. Testing / visualization / Final evaluation - compute metrics (pseudo masks compared with predicted masks)
    evaluate_seg_val_loader(seg_model,seg_val_loader, device)

    # Suppose you want to visualize indices 0, 1, and 2 from the val dataset (Image | Pseudo Mask | Predicted Mask)
    visualize_segmentation_val_ds(seg_model=seg_model,seg_val_ds=seg_val_ds,
                                          device=device,
                                          mean=(0.485, 0.456, 0.406),  # typical ImageNet
                                          std=(0.229, 0.224, 0.225),
                                          indices_to_show=[2, 5, 8]    # pick any indices you like
                                          )

    # Evaluate on Test dataset - compute metrics (by converting GT trimaps to binary and compare against predicted mask)
    # and visualize some predictions - this function calls -> evaluate_segmentation_metrics()
    evaluate_on_test()

    print("Workflow complete!")



if __name__ == "__main__":
    main()
