# Lecture-Aligned Part C Ready Summary

## 1. Task Type

Image classification

## 2. Experimental Settings

The experiments were conducted on the PKLot dataset for binary image classification of parking-space occupancy (occupied vs empty). Images were resized to 128x128 and normalized before training. The data split used 70% training, 15% validation, and 15% testing with a fixed random seed for reproducibility. Three lecture-aligned CNN architectures were trained from scratch: LeNet-5 CNN, AlexNet CNN, and ResNet-18 CNN. The previous custom CNN results were preserved separately in legacy_custom_cnn_results.csv and legacy_custom_cnn_part_c_ready.md. Models used PyTorch with the Adam optimizer, binary cross-entropy via BCEWithLogitsLoss, a batch size of 512, and up to 15 epochs. Dropout was applied in models where configured, data augmentation was used where configured, and early stopping was enabled according to the experiment configuration. Dataset usage: full PKLot. Total processed crops: 686084 (350349 empty, 335735 occupied). Class weighting: pos_weight=1.0435 from train split.

## 3. Lecture-Aligned Initial Results

| Model | Train Accuracy | Validation Accuracy | Test Accuracy |
| --- | ---: | ---: | ---: |
| LeNet-5 CNN | 0.9981 | 0.9981 | 0.9976 |
| AlexNet CNN | 0.9985 | 0.9984 | 0.9983 |
| ResNet-18 CNN | 0.9993 | 0.9993 | 0.9992 |
