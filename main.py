import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms


# A minimal Dataset to load images and labels from an annotation file.
class SimpleDataset(Dataset):
    def __init__(self, annotation_file, images_dir, transform=None):
        self.transform = transform
        self.data = []
        with open(annotation_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    image_name, class_id, _ = parts[:3]  # ignoring species_id
                    image_path = os.path.join(images_dir, f"{image_name}.jpg")
                    self.data.append((image_path, int(class_id)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    # Define directories and file paths.
    images_dir = "./images"
    trainval_file = "./annotations/trainval.txt"
    test_file = "./annotations/test.txt"

    # Define a simple transformation.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the trainval and test datasets.
    full_dataset = SimpleDataset(trainval_file, images_dir, transform)
    test_dataset = SimpleDataset(test_file, images_dir, transform)

    # Create a simple train/validation split (80% train, 20% validation).
    num_samples = len(full_dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_idx = int(0.8 * num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Build train and validation data lists.
    train_data = [full_dataset[i] for i in train_indices]
    val_data = [full_dataset[i] for i in val_indices]

    # Create DataLoaders.
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Number of training samples:", len(train_data))
    print("Number of validation samples:", len(val_data))
    print("Number of test samples:", len(test_dataset))


if __name__ == '__main__':
    main()


