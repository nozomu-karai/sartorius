import os
import time
import random
import collections
from logging import getLogger, FileHandler, StreamHandler, INFO, DEBUG, Formatter
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
from utils import fix_all_seeds, rle_encoding, remove_overlapping_pixels, MlflowWriter
from utils import combine_masks, get_filtered_masks, iou_map

import hydra
import warnings
warnings.filterwarnings('ignore')
logger = getLogger(__name__)
fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

def evaluate(cfg, model, ds, device):
    model.eval()
    iouscore = 0
    for i in tqdm(range(len(ds))):
        img, targets = ds[i]
        with torch.no_grad():
            result = model([img.to(device)])[0]

        masks = combine_masks(cfg, targets['masks'], 0.5)
        labels = pd.Series(result['labels'].cpu().numpy()).value_counts()

        mask_threshold = cfg.test.mask_threshold
        pred_masks = combine_masks(cfg, get_filtered_masks(cfg, result), mask_threshold)
        iouscore += iou_map([masks],[pred_masks])
    return iouscore / len(ds)

@hydra.main(config_name="config.yaml")
def main(cfg):
    cwd = hydra.utils.get_original_cwd()
    cfg.data.output_dir = os.path.join(cwd, cfg.data.output_dir)
    cfg.data.train_csv = os.path.join(cwd, cfg.data.train_csv)
    cfg.data.train_path = os.path.join(cwd, cfg.data.train_path)
    cfg.data.test_path = os.path.join(cwd, cfg.data.test_path)
    if not os.path.exists(cfg.data.output_dir):
        os.makedirs(cfg.data.output_dir)
    fix_all_seeds(2021)

    if cfg.do_train:
        fh = FileHandler(os.path.join(cfg.data.output_dir, "train.log"), 'w')
    if cfg.do_eval:
        fh = FileHandler(os.path.join(cfg.data.output_dir, "eval.log"), 'w')
    fh.setLevel(INFO)
    fh.setFormatter(fmr)
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))
    logger.info(cfg)

    logger.info("*****data set*****")
    df_base = pd.read_csv(cfg.data.train_csv, nrows=5000 if cfg.test.test else None)
    df_images = df_base.groupby(["id", "cell_type"]).agg({'annotation': 'count'}).sort_values("annotation", ascending=False).reset_index()
    df_images_train, df_images_val = train_test_split(df_images, stratify=df_images['cell_type'], 
                                                  test_size=cfg.data.val_ratio)
    df_train = df_base[df_base['id'].isin(df_images_train['id'])]
    df_valid = df_base[df_base['id'].isin(df_images_val['id'])]
    ds_train = CellDataset(cfg.data.train_path, df_train, resize=False, transforms=get_transform(train=True, cfg=cfg), cfg=cfg)
    ds_valid = CellDataset(cfg.data.train_path, df_valid, resize=False, transforms=get_transform(train=False, cfg=cfg), cfg=cfg)
    dl_train = DataLoader(ds_train, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    n_batches = len(dl_train)
    height = ds_train.height
    width = ds_train.width

    logger.info(f"step_size: {n_batches}")
    logger.info(f"[Train]  # of picture: {len(df_images_train)}, # of instance: {len(df_train)}")
    logger.info(f"[Valid]  # of picture: {len(df_images_val)}, # of instance: {len(df_valid)}")
    logger.info(f"width: {width}, height: {height}")

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=cfg.model.box_detections_per_img)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.model.num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, cfg.model.num_classes)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for param in model.parameters():
        param.requires_grad = True

    if cfg.do_train:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=cfg.train.learning_rate, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        logger.info("*****Training*****")
        best_score = None
        for epoch in range(1, cfg.train.num_epochs + 1):
            model.train()
            logger.info(f"Starting epoch {epoch} of {cfg.train.num_epochs}")

            time_start = time.time()
            loss_accum = 0.0
            loss_mask_accum = 0.0

            train_bar = tqdm(dl_train)
            for batch_idx, (images, targets) in enumerate(train_bar, 1):

                # Predict
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                if n_gpu > 1:
                    loss = loss.mean()

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

            prefix = f"[Epoch {epoch:2d} / {cfg.train.num_epochs:2d}]"
            logger.info(f"{prefix} Train mask-only loss: {train_loss_mask:7.3f}")
            logger.info(f"{prefix} Train loss: {train_loss:7.3f}. [{elapsed:.0f} secs]")

            score = evaluate(cfg, model, ds_valid, device)
            logger.info(f"valid IoU score: {score}")
            if (best_score is None) or score > best_score: 
                logger.info(f"best model saved [pytorch_model-e{epoch}.bin], IoU score: {score}")
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(cfg.data.output_dir, "pytorch_model-best.bin"))
                best_score = score

        ds_test = CellTestDataset(cfg.data.test_path, transforms=get_transform(train=False, cfg=cfg))

        model.eval()

        logger.info("*****Prediction*****")
        submission = []
        for sample in ds_test:
            img = sample['image']
            image_id = sample['image_id']
            with torch.no_grad():
                result = model([img.to(device)])[0]

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

        writer = MlflowWriter("sartorisu")
        writer.log_params_from_omegaconf_dict(cfg)
        writer.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
        writer.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
        writer.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
        writer.log_artifact(os.path.join(os.getcwd(), 'run.log'))

    if cfg.do_eval:
        param = torch.load(os.path.join(cfg.data.output_dir, "pytorch_model-best.bin"))
        model.load_state_dict(param)
        logger.info(f"model weight loaded from [pytorch_model-e1.bin]")
        score = evaluate(cfg, model, ds_valid, device)
        logger.info(f"valid IoU score: {score}")

if __name__ == "__main__":
    main()
