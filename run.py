import os
import time
import random
import collections
from logging import getLogger, FileHandler, basicConfig, INFO
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset import CellDataset, CellTestDataset, get_transform
from utils import fix_all_seeds, rle_encoding, remove_overlapping_pixels

import hydra
import mlflow
import warnings
warnings.filterwarnings('ignore')
basicConfig(format='[%(levelname)s] %(asctime)s >>\t%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=INFO)
logger = getLogger(__name__)

@hydra.main(config_name="config.yaml")
def main(cfg):
    cwd = hydra.utils.get_original_cwd()
    cfg.data.output_dir = os.path.join(cwd, cfg.data.output_dir)
    fix_all_seeds(2021)
    logger.addHandler(FileHandler(os.path.join(cfg.data.output_dir, "train.log"), 'w'))

    df_train = pd.read_csv(cfg.data.train_csv, nrows=5000 if cfg.test.test else None)
    ds_train = CellDataset(cfg.data.train_path, df_train, resize=False, transforms=get_transform(train=True))
    dl_train = DataLoader(ds_train, batch_size=cfg.train.batch_size, shuffle=True, 
                      num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=cfg.model.box_detections_per_img)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.model.num_classes)

    model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.train.learning_rate, momentum=cfg.train.momentun, weight_decay=cfg.train.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    n_batches = len(dl_train)

    for epoch in range(1, cfg.train.num_epochs + 1):
        logger.info(f"Starting epoch {epoch} of {cfg.train.num_epochs}")
        
        time_start = time.time()
        loss_accum = 0.0
        loss_mask_accum = 0.0

        train_bar = tqdm(dl_train)
        for batch_idx, (images, targets) in enumerate(train_bar, 1):
        
            # Predict
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            loss_mask = loss_dict['loss_mask'].item()
            loss_accum += loss.item()
            loss_mask_accum += loss_mask

            train_bar.set_description(f"[Epoch {epoch:2d} / {cfg.train.num_epochs:2d}] Batch train loss: {loss.item():7.3f}. Mask-only loss: {loss_mask:7.3f}")
        
        if cfg.train.use_scheduler:
            lr_scheduler.step()
        
        # Train losses
        train_loss = loss_accum / n_batches
        train_loss_mask = loss_mask_accum / n_batches
        
        elapsed = time.time() - time_start
        
        torch.save(model.state_dict(), f"pytorch_model-e{epoch}.bin")
        prefix = f"[Epoch {epoch:2d} / {cfg.train.num_epochs:2d}]"
        logger.info(f"{prefix} Train mask-only loss: {train_loss_mask:7.3f}")
        logger.info(f"{prefix} Train loss: {train_loss:7.3f}. [{elapsed:.0f} secs]")

    ds_test = CellTestDataset(cfg.data.test_path, transforms=get_transform(train=False))

    model.eval()

    submission = []
    for sample in ds_test:
        img = sample['image']
        image_id = sample['image_id']
        with torch.no_grad():
            result = model([img.to(DEVICE)])[0]
        
        previous_masks = []
        for i, mask in enumerate(result["masks"]):
            
            # Filter-out low-scoring results. Not tried yet.
            score = result["scores"][i].cpu().item()
            if score < cfg.test.min_score:
                continue
            
            mask = mask.cpu().numpy()
            # Keep only highly likely pixels
            binary_mask = mask > cfg.test.mask_threshold
            binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
            previous_masks.append(binary_mask)
            rle = rle_encoding(binary_mask)
            submission.append((image_id, rle))
        
        # Add empty prediction if no RLE was generated for this image
        all_images_ids = [image_id for image_id, rle in submission]
        if image_id not in all_images_ids:
            submission.append((image_id, ""))

    df_sub = pd.DataFrame(submission, columns=['id', 'predicted'])
    df_sub.to_csv(os.path.join(cfg.data.output_dir, "submission.csv"), index=False)

if __name__ == "__main__":
    main()