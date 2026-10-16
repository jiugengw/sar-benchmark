"""Labeled course training data: every image has one class label and >= 1 box.
Loads images and boxes at a fixed input size and maps predicted boxes back to original pixels."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

# Mean and std of all pixels of the 9,392 train images, read as gray, resized to
# 512 x 512 and scaled to [0, 1] (computed once).
MEAN, STD = 0.2526, 0.2133


def padded_size(orig_size, size=512):
    h, w = orig_size
    scale = size / max(h, w)
    return max(1, round(h * scale)), max(1, round(w * scale))


class ResizePad:
    """Resize without stretching, then pad on the right and bottom (also transforms boxes)."""

    def __init__(self, size=512):
        self.size = size

    def __call__(self, image, boxes=None):
        from torchvision.transforms.v2 import functional as TF
        h, w = padded_size(image.shape[-2:], self.size)
        image = TF.pad(TF.resize(image, [h, w]), [0, 0, self.size - w, self.size - h])
        if boxes is None:
            return image
        boxes = TF.pad(TF.resize(boxes, [h, w]), [0, 0, self.size - w, self.size - h])
        return image, boxes


class SARMultiTask(Dataset):
    """Images from train/labels.csv with boxes from train/instances.json under the data root.
    An item is (image [1, size, size], label 0..8, target), target = {'boxes': XYXY on the resized
    image, 'labels': 1..9, 'image_id': COCO id, 'orig_size': (h, w)}."""

    def __init__(self, root, train=False, size=512, limit=None, indices=None, geometry='stretch',
                 photometric=False, small_object_sampling=False):
        self.root = Path(root)
        self.geometry = geometry
        self.image_root = self.root / 'train'
        with open(self.image_root / 'labels.csv', newline='') as f:
            self.rows = list(csv.DictReader(f))[:limit]
        if indices is not None:
            self.rows = [self.rows[i] for i in indices]
        coco = json.loads((self.image_root / 'instances.json').read_text())
        self.coco_images = {img['file_name']: img for img in coco['images']}
        self.annotations = defaultdict(list)  # COCO image id -> its boxes
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        # Count boxes that remain tiny at the 512 px training resolution. This is
        # used only by the optional image sampler; labels and boxes are untouched.
        self.small_object_counts = None
        if small_object_sampling:
            self.small_object_counts = []
            for row in self.rows:
                info = self.coco_images[row['file_name']]
                scale_x = size / info['width'] if geometry == 'stretch' else min(size / info['width'], size / info['height'])
                scale_y = size / info['height'] if geometry == 'stretch' else scale_x
                self.small_object_counts.append(sum(
                    (ann['bbox'][2] * scale_x) * (ann['bbox'][3] * scale_y) < 16 ** 2
                    for ann in self.annotations[info['id']]))
        flips = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] if train else []
        appearance = [v2.ColorJitter(brightness=0.15, contrast=0.15)] if train and photometric else []
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat('XYXY'),  # COCO [x, y, w, h] -> [x1, y1, x2, y2]
            ResizePad(size) if geometry == 'pad' else v2.Resize((size, size)),
            *flips,  # overhead imagery has no canonical "up"
            v2.ToDtype(torch.float32, scale=True),
            *appearance,
            v2.Normalize([MEAN], [STD]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        info = self.coco_images[row['file_name']]
        anns = self.annotations[info['id']]
        orig_size = (info['height'], info['width'])
        boxes = tv_tensors.BoundingBoxes([a['bbox'] for a in anns], format='XYWH', canvas_size=orig_size,
                                         dtype=torch.float32)  # integer boxes would be truncated by Resize
        image = Image.open(self.image_root / row['file_name']).convert('L')  # PIL: torchvision.io cannot read .bmp
        image, boxes = self.transform(image, boxes)
        target = {'boxes': boxes, 'labels': torch.tensor([a['category_id'] for a in anns]),
                  'image_id': info['id'], 'orig_size': orig_size, 'geometry': self.geometry}
        return image, int(row['label']), target


def collate(batch):
    """Stack images and labels; targets stay a list because images have different numbers of boxes."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


def to_original_xywh(boxes_xyxy, orig_size, size=512, geometry='stretch'):
    """Undo the resize: XYXY boxes on the size x size input -> COCO [x, y, w, h] in original pixels."""
    h, w = orig_size
    rh, rw = padded_size(orig_size, size) if geometry == 'pad' else (size, size)
    boxes = boxes_xyxy * boxes_xyxy.new_tensor([w / rw, h / rh, w / rw, h / rh])
    if geometry == 'pad':
        boxes = torch.stack([boxes[:, 0].clamp(0, w), boxes[:, 1].clamp(0, h),
                             boxes[:, 2].clamp(0, w), boxes[:, 3].clamp(0, h)], dim=1)
    return box_convert(boxes, 'xyxy', 'xywh')
