import torchvision
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader

root_dir = "./oxford_iiit_data"  # or any folder you choose

dummy_dataload = torchvision.datasets.OxfordIIITPet(
    root=root_dir,
    split="trainval",
    target_types=["category"],  # or ["segmentation"] / ["category", "segmentation"]
    download=True,
)

def parse_xml_for_bbox(xml_file):
    """Parse the Oxford-IIIT Pet XML file for bounding box [xmin, ymin, xmax, ymax]."""
    if not os.path.exists(xml_file):
        # Return None or raise an exception.
        # For skipping, you could do:
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
    """
    Reads lines from trainval.txt or test.txt.
    Returns a list of image IDs (strings).
    """
    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n')
    # Each line might look like "Abyssinian_1 1 1"
    # We'll take the first token as the image base name and 2nd token as breed-class
    name= [line.split()[0] for line in lines if line.strip()]
    breed= [line.split()[1] for line in lines if line.strip()]
    return name, breed

def build_annotations_list(root_dir, txt_filename):
    """
    Build a list of annotation dicts from trainval.txt or test.txt.
    Each dict might look like:
        {
          'image_id': 'Abyssinian_1.jpg',
          'boxes': [[xmin, ymin, xmax, ymax]],    # can have multiple
          'labels': ['Abyssinian']                # or numeric label if you prefer
        }
    """
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
            # Means parse_xml_for_bbox returned None -> skip
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
    """
    Loads 'trainval.txt' and 'test.txt' from the dataset folder, returning
    two lists of annotation dicts.
    """
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
        labels = ann['labels']  # List of label strings

        # Load image as a NumPy array.
        image = np.array(Image.open(img_path).convert('RGB'))

        # Duplicate labels for each bounding box to match the length of bboxes
        labels = [labels] * len(boxes)  # This now creates a list of labels


        if self.transform:
            # Albumentations expects keys: image, bboxes, labels.
            # It will output a dict with keys: image, bboxes, labels.
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']



        # Convert bounding boxes to a tensor.
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        # pick from list, change type to int and -1 so that [1..37] --> [0..36] class labels
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
    """
    Custom collate function for object detection.

    Args:
        batch: list of tuples (image_tensor, target_dict)

    Returns:
        batch_images: Tensor [B, C, H, W]
        batch_targets: list of target dicts (each with 'boxes' and 'labels')
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    paths = [item[2] for item in batch]
    images = torch.stack(images, dim=0)  # This works because all images are resized

    return images, targets, paths

def get_dataloader():
    trainval_anns = load_oxford_pet_annotations("/content/oxford_iiit_data/oxford-iiit-pet")
    trainval_dataset = OxfordPetBboxDatasetAlbumentations(trainval_anns, transform=albumentations_transform)

    img_paths = []
    for i in range(len(trainval_anns)):
        img_paths.append(trainval_anns[i]['image_file'])

    # DataLoader
    train_loader = DataLoader(trainval_dataset, batch_size=8, shuffle=False,  num_workers=2, collate_fn=detection_collate_fn)
    return train_loader, img_paths