import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from dataset import MyDataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet34, ResNet34_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

torch.set_float32_matmul_precision('high')
class Collaborative(pl.LightningModule):
    def __init__(self, num_classes=4, class_weight=1.0, bbox_weight=2.0, max_objects=6, confidence_threshold=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion_box= nn.SmoothL1Loss()
        self.max_objects = max_objects
        self.num_classes = num_classes

        self.model = resnet34(weights = ResNet34_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(num_features, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.num_classes)
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(num_features, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.max_objects * 4)
        )

        self.confidence_predictor = nn.Sequential(
            nn.Linear(num_features, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.max_objects)
        )

        clf_metrics_args = {'num_labels': self.num_classes, 'threshold': self.hparams.confidence_threshold}
        self.val_f1_macro = torchmetrics.classification.MultilabelF1Score(**clf_metrics_args, average='macro')
        self.val_hamming = torchmetrics.classification.MultilabelHammingDistance(**clf_metrics_args)

        conf_metrics_args = {'threshold': self.hparams.confidence_threshold}
        self.val_conf_f1 = torchmetrics.classification.BinaryF1Score(**conf_metrics_args)
        self.val_conf_acc = torchmetrics.classification.BinaryAccuracy(**conf_metrics_args)
        self.val_count_mae = torchmetrics.regression.MeanAbsoluteError()

        self.val_avg_iou = torchmetrics.MeanMetric()

        self.confidence_threshold = confidence_threshold

    def forward(self, img):
        x = self.model(img)

        class_preds = self.classifier(x)
        bbox_preds = self.bbox_regressor(x).view(-1, self.max_objects, 4)
        confidence_preds = self.confidence_predictor(x).view(-1, self.max_objects)

        return class_preds, bbox_preds, confidence_preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.layer3.parameters(), 'lr': 1e-4},
            {'params': self.model.layer4.parameters(), 'lr': 1e-4},
            {'params': self.classifier.parameters(), 'lr': 1e-3},
            {'params': self.bbox_regressor.parameters(), 'lr': 1e-3},
            {'params': self.confidence_predictor.parameters(), 'lr': 1e-3}
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
        img, box_label, img_label = batch
        batch_size = img.shape[0]
        indexes_box = torch.any(box_label >= 0, dim=2)
        class_preds, bbox_preds, confidence_preds = self(img)

        loss_class = 0
        loss_box = 0
        loss_confidence = 0

        for i in range(batch_size):
            current_mask = indexes_box[i]
            if torch.any(current_mask):
                loss_class += self.criterion(class_preds[i], img_label[i])

                valid_boxes_pred = bbox_preds[i][current_mask]
                valid_boxes_target = box_label[i][current_mask]
                loss_box += self.criterion_box(valid_boxes_pred, valid_boxes_target)

                target_confidence = torch.zeros_like(confidence_preds[i])
                target_confidence[current_mask] = 1.0
                loss_confidence += self.criterion(confidence_preds[i], target_confidence)

                if len(valid_boxes_pred) > 0 and len(valid_boxes_target) > 0:
                    iou = torchvision.ops.box_iou(valid_boxes_pred, valid_boxes_target)
                    best_iou_values, _ = iou.max(dim=0)
                    iou_loss = 1 - best_iou_values.mean()
                    loss_box += iou_loss

        loss_class /= batch_size
        loss_box /= batch_size
        loss_confidence /= batch_size

        loss = (self.class_weight * loss_class) + (self.bbox_weight * loss_box) + loss_confidence
        self.log(f"{step_type}_loss_total", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_loss_segmentation", loss_box, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_loss_classification", loss_class, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}loss_confidence", loss_confidence, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if step_type == 'val':
            self.metric_step(class_preds, confidence_preds, bbox_preds, img_label, box_label)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def metric_step(self, class_preds, confidence_preds, bbox_preds, img_label, box_label):
        target_labels_int = img_label.int()
        class_probs = torch.sigmoid(class_preds)
        self.val_f1_macro.update(class_probs, target_labels_int)
        self.val_hamming.update(class_probs, target_labels_int)

        self.log('val_F1_macro', self.val_f1_macro, on_step=False, on_epoch=True,prog_bar=True, logger=True)
        self.log('val_Hamming', self.val_hamming, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        conf_probs = torch.sigmoid(confidence_preds)
        target_confidence_binary = torch.any(box_label >= 0, dim=2).int()
        flat_conf_probs = conf_probs.flatten()
        flat_target_conf = target_confidence_binary.flatten()

        self.val_conf_f1.update(flat_conf_probs, flat_target_conf)
        self.val_conf_acc.update(flat_conf_probs, flat_target_conf)

        self.log('val_Conf_F1', self.val_conf_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_Conf_Acc', self.val_conf_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        pred_counts = (conf_probs > self.confidence_threshold).sum(dim=1)
        true_counts = target_confidence_binary.sum(dim=1)

        self.val_count_mae.update(pred_counts, true_counts)

        self.log('val_Count_MAE', self.val_count_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        batch_size = class_preds.shape[0]
        for i in range(batch_size):
            gt_boxes_i = box_label[i]
            pred_boxes_i = bbox_preds[i]
            pred_confs_i = conf_probs[i]

            gt_mask_i = torch.any(gt_boxes_i >= 0, dim=1)
            valid_gt_boxes = gt_boxes_i[gt_mask_i]
            num_valid_gt = valid_gt_boxes.shape[0]
            if num_valid_gt == 0:
                continue

            pred_mask_i = pred_confs_i >= self.confidence_threshold
            valid_pred_boxes = pred_boxes_i[pred_mask_i]
            num_valid_pred = valid_pred_boxes.shape[0]

            if num_valid_pred == 0:
                mean_best_iou_image = 0.0
            else:
                try:
                    iou_matrix = torchvision.ops.box_iou(valid_pred_boxes, valid_gt_boxes)
                except Exception as e:
                    print(f"Warning: IoU calculation failed for image {i} in batch: {e}")
                    mean_best_iou_image = 0.0
                    iou_matrix = None

                if iou_matrix is not None and iou_matrix.numel() > 0:
                    best_iou_per_gt, _ = iou_matrix.max(dim=0)
                    mean_best_iou_image = best_iou_per_gt.mean().item()

                elif iou_matrix is not None and iou_matrix.numel() == 0 and num_valid_pred > 0:
                    mean_best_iou_image = 0.0

            self.val_avg_iou.update(mean_best_iou_image, weight=num_valid_gt)

        self.log('val_AvgBestIoU', self.val_avg_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        class_names = ['bike', 'cars', 'motorbikes', 'person']
        img, boxes = batch

        class_preds, bbox_preds, confidence_preds = self(img)
        class_preds = torch.sigmoid(class_preds) > 0.5
        confidence_preds = torch.sigmoid(confidence_preds) > 0.5
        filtered_boxes_preds = bbox_preds[confidence_preds]
        count_pred_boxes = int(confidence_preds.sum())
        real_boxes = boxes >= 0
        count_real_boxes = int(real_boxes.sum()/4)

        predicted_classes = [class_names[i] for i, pred in enumerate(class_preds[0]) if pred]
        if not predicted_classes:
            predicted_classes = 'not detected'

        text = f'number of objects: {count_pred_boxes}/{count_real_boxes}, class preds: {predicted_classes}'
        fig, ax = plt.subplots()
        ax.text(0, -20, text, color="white",
                fontsize=12, bbox=dict(facecolor="black", alpha=0.7))

        img.squeeze_()
        img_numpy = img.permute(1, 2, 0).cpu().numpy()
        img_numpy = (img_numpy * std) + mean
        img_numpy = (img_numpy * 255).astype(np.uint8)
        img_display = torch.from_numpy(img_numpy).permute(2, 0, 1)
        try:
            img_display = torchvision.utils.draw_bounding_boxes(img_display, filtered_boxes_preds, colors='red', width=1)
            complete_img = torchvision.utils.draw_bounding_boxes(img_display, boxes[0], colors='green', width=1)

        except Exception as e:
            complete_img = img
            print(f'error: {e}')

        complete_img = complete_img.permute(1, 2, 0).cpu().numpy()

        plt.imshow(complete_img)
        plt.axis('off')
        plt.show()



if __name__ == '__main__':
    traindataset = MyDataset('data/train_png', True, 'train')
    traindataloader = DataLoader(traindataset, batch_size = 32, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)

    valdatset = MyDataset('data/val_png', False, 'val')
    valdataloader = DataLoader(valdatset, batch_size = 32, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)

    testdataset = MyDataset('data/test_png', False, 'test')
    testtdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    loaded_model = Collaborative()

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/without_mask',
        filename='model{i}-{epoch:02d}-{val_loss_total:.2f}',
        save_top_k=1,
        monitor='val_loss_total',
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(max_epochs=100, fast_dev_run=False, accelerator='gpu', log_every_n_steps=10, callbacks=[checkpoint_callback])
    trainer.fit(loaded_model, traindataloader, valdataloader)
    trainer.predict(loaded_model, testtdataloader)

