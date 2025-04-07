import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from dataset_mask import MyDataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet34, ResNet34_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint
import segmentation_models_pytorch as smp
import cv2
import torchmetrics

torch.set_float32_matmul_precision('high')
class Collaborative(pl.LightningModule):
    def __init__(self, confidence_threshold = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.BCEWithLogitsLoss()

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        self.model_classifier = resnet34(weights = ResNet34_Weights.DEFAULT)
        self.model_classifier.fc = nn.Linear(512, 4)


        for param in self.model.encoder.parameters():
            param.requires_grad = False

        for param in self.model.encoder.layer3.parameters():
            param.requires_grad = True

        for param in self.model.encoder.layer4.parameters():
            param.requires_grad = True

        for param in self.model_classifier.parameters():
            param.requires_grad = False

        for param in self.model_classifier.layer4.parameters():
            param.requires_grad = True

        for param in self.model_classifier.fc.parameters():
            param.requires_grad = True

        clf_metrics_args = {'num_labels': 4, 'threshold': self.hparams.confidence_threshold}
        self.val_f1_macro = torchmetrics.classification.MultilabelF1Score(**clf_metrics_args, average='macro')
        self.val_hamming = torchmetrics.classification.MultilabelHammingDistance(**clf_metrics_args)

        conf_metrics_args = {'threshold': self.hparams.confidence_threshold}
        self.val_count_mae = torchmetrics.regression.MeanAbsoluteError()

        self.val_avg_iou = torchmetrics.MeanMetric()

        self.confidence_threshold = confidence_threshold

    def forward(self, img):
        return self.model(img)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.encoder.layer3.parameters(), 'lr': 1e-4},
            {'params': self.model.encoder.layer4.parameters(), 'lr': 1e-4},
            {'params': self.model.decoder.parameters(), 'lr': 1e-3},
            {'params': self.model_classifier.layer4.parameters(), 'lr': 1e-4},
            {'params': self.model_classifier.fc.parameters(), 'lr': 1e-3}
        ])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss_total'
            }
        }

    def _shared_step(self, batch, batch_idx, step_type):
        imgs, masks, box_label, annotations = batch
        pixel_preds = self(imgs)
        pixel_preds.squeeze_()
        loss_object = self.criterion(pixel_preds, masks)
        object_class = self.model_classifier(imgs)
        loss_class = self.criterion(object_class, annotations)
        loss = loss_object + loss_class * 2
        self.log(f"{step_type}_loss_total", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_loss_segmentation", loss_object, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_loss_classification", loss_class, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if step_type == 'val':
            self.metric_step(object_class, pixel_preds, annotations, box_label)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def metric_step(self, class_preds, pixel_preds, img_label, box_label):
        target_labels_int = img_label.int()
        class_probs = torch.sigmoid(class_preds)

        self.val_f1_macro.update(class_probs, target_labels_int)
        self.val_hamming.update(class_probs, target_labels_int)
        self.log('val_F1_macro', self.val_f1_macro, on_step=False, on_epoch=True,prog_bar=True, logger=True)
        self.log('val_Hamming', self.val_hamming, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        num_of_all_objects = []
        num_of_all_bbox_preds = []
        for i in range(0, pixel_preds.shape[0]):
            bbox_preds, count_object_pred = self.count_objects(pixel_preds[i,:,:])
            num_of_all_objects.append(count_object_pred)
            num_of_all_bbox_preds.append(bbox_preds)


        target_confidence_binary = torch.any(box_label >= 0, dim=2).int()
        true_counts = target_confidence_binary.sum(dim=1)
        num_of_all_objects = torch.tensor(num_of_all_objects, device='cuda:0')

        self.val_count_mae.update(num_of_all_objects, true_counts)
        self.log('val_Count_MAE', self.val_count_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        batch_size = class_preds.shape[0]
        for i in range(batch_size):
            gt_boxes_i = box_label[i]
            pred_boxes_i = num_of_all_bbox_preds[i]

            gt_mask_i = torch.any(gt_boxes_i >= 0, dim=1)
            valid_gt_boxes = gt_boxes_i[gt_mask_i]

            num_valid_gt = valid_gt_boxes.shape[0]
            if num_valid_gt == 0:
                continue

            else:
                try:
                    iou_matrix = torchvision.ops.box_iou(pred_boxes_i, valid_gt_boxes)
                except Exception as e:
                    print(f"Warning: IoU calculation failed for image {i} in batch: {e}")
                    mean_best_iou_image = 0.0
                    iou_matrix = None

                if iou_matrix is not None and iou_matrix.numel() > 0:
                    best_iou_per_gt, _ = iou_matrix.max(dim=0)
                    mean_best_iou_image = best_iou_per_gt.mean().item()

            self.val_avg_iou.update(mean_best_iou_image, weight=num_valid_gt)

        self.log('val_AvgBestIoU', self.val_avg_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def predict_step(self, batch, batch_idx):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        class_names = ['bike', 'cars', 'motorbikes', 'person']
        img, boxes = batch
        pixel_preds = self(img)
        boxes_tenor_pred, count_object_pred = self.count_objects(pixel_preds)
        real_boxes = boxes >= 0
        count_real_boxes = int(real_boxes.sum()/4)

        class_preds = self.model_classifier(img)
        class_preds = torch.sigmoid(class_preds) > 0.8
        predicted_classes = [class_names[i] for i, pred in enumerate(class_preds[0]) if pred]
        if not predicted_classes:
            predicted_classes = 'not detected'

        text = f'number of objects: {count_object_pred}/{count_real_boxes}, class preds: {predicted_classes}'
        fig, ax = plt.subplots()
        ax.text(0, -20, text, color="white",
                fontsize=12, bbox=dict(facecolor="black", alpha=0.7))

        img.squeeze_()
        img_numpy  = img.permute(1, 2, 0).cpu().numpy()
        img_numpy  = (img_numpy  * std) + mean
        img_numpy  = (img_numpy  * 255).astype(np.uint8)
        img_display = torch.from_numpy(img_numpy).permute(2, 0, 1)
        try:
            img_display = torchvision.utils.draw_bounding_boxes(img_display, boxes_tenor_pred, colors='red', width=1)
            complete_img = torchvision.utils.draw_bounding_boxes(img_display, boxes[0], colors='green', width=1)

        except Exception as e:
            complete_img = img
            print(f'error: {e}')

        complete_img = complete_img.permute(1, 2, 0).cpu().numpy()

        plt.imshow(complete_img)
        plt.axis('off')
        plt.show()

    def count_objects(self, mask_pred):
        mask_pred  = torch.sigmoid(mask_pred)
        preds_binary = (mask_pred > 0.5).float()

        masks = preds_binary.cpu().numpy().astype(np.uint8)
        masks = masks.squeeze()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(masks)

        boxes = []
        min_pixels = 100
        count_objects = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_pixels:
                count_objects += 1
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                x2 = x + w
                y2 = y + h

                boxes.append([x, y, x2, y2])
        boxes_tenor = torch.tensor(boxes, dtype=torch.float32, device='cuda:0')
        return boxes_tenor, count_objects


if __name__ == '__main__':
    traindataset = MyDataset('data/train_png', True, 'train')
    traindataloader = DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)

    valdatset = MyDataset('data/val_png', False, 'val')
    valdataloader = DataLoader(valdatset, batch_size=32, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)

    testdataset = MyDataset('data/test_png', False, 'test')
    testtdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    loaded_model = Collaborative()

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/with_mask',
        filename='model{i}-{epoch:02d}-{val_loss_total:.2f}',
        save_top_k=1,
        monitor='val_loss_total',
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(max_epochs=100, fast_dev_run=False, accelerator='gpu', log_every_n_steps=10, callbacks=[checkpoint_callback])
    trainer.fit(loaded_model, traindataloader, valdataloader)
    trainer.predict(loaded_model, testtdataloader)
