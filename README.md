# Object Classification and Detection Models (PASCAL VOC 2005)

This repository contains two models based on PyTorch Lightning, designed for multi-label image classification and object detection for 4 classes: 'bike', 'cars', 'motorbikes', 'person'. Both models were trained on data derived from the PASCAL VOC 2005 dataset (224x224 px images), utilizing different approaches to the detection task.

## Models

### 1. Direct Detection Model

* **Files:** `main.py`, `dataset.py`
* **Approach:**
    * The model uses a **ResNet34** architecture (pre-trained on ImageNet) as the base feature extractor. Early layers are frozen, while `layer3` and `layer4` are fine-tuned.
    * It features three separate output heads attached to the ResNet34 features:
        1.  For **multi-label image classification** (predicting the presence of each of the 4 classes).
        2.  For **bounding box regression** (directly predicting coordinates for up to 6 objects).
        3.  For **confidence prediction** (estimating whether a predicted object slot actually contains an object).
    * Detection is performed through direct prediction of coordinates and confidence scores for potential objects.
* **Framework / Libraries:** PyTorch, PyTorch Lightning, Torchvision, Torchmetrics, Albumentations, OpenCV.
* **Input (Training):** Images (224x224), image-level class labels, ground truth bounding box coordinates.
* **Output:** Probabilities for each of the 4 classes, coordinates for up to 6 bounding boxes, confidence scores for each box.

### 2. Segmentation-based Detection Model

* **Files:** `main_mask.py`, `dataset_mask.py`
* **Approach:**
    * The model consists of two main components:
        1.  A **U-Net** (from the `segmentation-models-pytorch` library) with a **ResNet34** encoder for **binary semantic segmentation** (separating all objects from the background).
        2.  A separate **ResNet34** model for **multi-label image classification** (4 classes).
    * **Object detection is achieved indirectly**:
        1.  The model predicts a segmentation mask.
        2.  Then, in a post-processing step, connected components analysis (`cv2.connectedComponentsWithStats`) is applied to the mask to find object regions and generate bounding boxes for them.
* **Framework / Libraries:** PyTorch, PyTorch Lightning, Torchvision, Torchmetrics, Segmentation Models Pytorch (SMP), Albumentations, OpenCV.
* **Input (Training):** Images (224x224), **ground truth segmentation masks**, image-level class labels, ground truth bounding box coordinates (for evaluation).
* **Output:** A binary segmentation mask, probabilities for each of the 4 classes. Bounding boxes are generated during the prediction/post-processing step.

## Performance Comparison (after 100 epochs)

The table below shows the validation metrics obtained for both models after 100 epochs of training. The `val_loss_total` value is not directly comparable due to different loss function definitions.

| Metric             | Model 1 (Direct Detection) | Model 2 (Segmentation-based) | Metric Description                       |
| :------------------ | :------------------------- | :--------------------------- | :--------------------------------------- |
| `val_F1_macro`      | 0.6195                     | **0.6274** | F1-Score (Macro Avg) for classification |
| `val_Hamming`       | 0.0757                     | **0.0731** | Hamming Distance for classification (lower is better) |
| `val_Count_MAE`     | **0.1640** | 0.2924                       | Mean Absolute Error for object counting (lower is better) |
| `val_AvgBestIoU`    | 0.5335                     | **0.6684** | Mean Best IoU (localization quality)     |
| `val_loss_total`    | (19.6715)                  | (0.6413)                     | (Not directly comparable)                |

**Comparison Conclusions:**

* **Classification:** Both models achieve very similar performance on the multi-label classification task. Model 2 (segmentation-based) shows slightly better results for both F1-macro and Hamming Distance.
* **Object Counting:** Model 1 (direct detection) is **significantly better** at predicting the correct number of objects in an image (lower MAE). This might be due to its dedicated confidence prediction head, as opposed to counting objects based on connected components in Model 2, which can be sensitive to mask splitting/merging.
* **Localization Quality (IoU):** Model 2 (segmentation-based) achieves a **significantly higher** Mean Best IoU score. This suggests that generating boxes based on predicted segmentation masks leads to more accurate object localization in this case.
* **Summary:** Neither model is definitively superior across all metrics.
    * **Model 1** is preferred if the priority is **accurate object counting**.
    * **Model 2** is a better choice if **high localization accuracy (IoU)** is crucial, at the cost of slightly less accurate counting.

## Prediction Examples
*Below is an example visualization of a prediction. Red boxes indicate the model's predictions, while green boxes represent the ground truth. The text includes the predicted object count and the predicted classes for the entire image.*

### Model 1 (Direct Detection)
![1](https://github.com/user-attachments/assets/60f3b87f-335d-4a22-b786-17d5412b1ee9)
![2](https://github.com/user-attachments/assets/daaa5d53-902c-4bda-af26-7931acd93b86)
![3](https://github.com/user-attachments/assets/4c20060a-0ca2-43b3-aabb-b7fda5d1e547)
![4](https://github.com/user-attachments/assets/d6d10a19-8951-437d-a542-7641c6c24d1e)
![5](https://github.com/user-attachments/assets/36a309dc-e851-403c-a080-2ee526fbad99)

### Model 2 (Segmentation-based Detection)
![1-w](https://github.com/user-attachments/assets/9e57b841-6352-4c83-8919-be21784a396a)
![2-w](https://github.com/user-attachments/assets/613a63b7-93b0-422b-93e2-2c986cda5e5b)
![3w](https://github.com/user-attachments/assets/bf809757-4702-4655-8efd-c62497c27d15)
![4w](https://github.com/user-attachments/assets/bf84301b-629d-408f-8736-ad7bff35ad9f)
![5w](https://github.com/user-attachments/assets/76cac43a-2365-450e-8c59-6586087469c7)
