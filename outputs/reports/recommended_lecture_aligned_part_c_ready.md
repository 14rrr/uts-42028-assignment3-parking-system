# Recommended Lecture-Aligned Part C Summary

## 1. Task Type

Image classification

## 2. Recommended Comparison Set

The recommended Part C comparison uses three lecture-aligned CNN architectures trained from scratch on the full processed PKLot binary parking-space dataset:

- LeNet-5 CNN
- AlexNet CNN
- ResNet-18 CNN

The previous custom CNN results were preserved separately as legacy outputs and were not deleted or overwritten.

## 3. Experimental Settings

Images were resized to 128x128 and normalized. The existing 70% training, 15% validation, and 15% test split was reused with seed 42. Training used the existing PyTorch pipeline with CUDA, AMP, channels-last memory format, Adam, BCEWithLogitsLoss, batch size 512, and up to 15 epochs with early stopping according to the model configuration.

## 4. Results

| Model | Train Accuracy | Validation Accuracy | Test Accuracy |
| --- | ---: | ---: | ---: |
| LeNet-5 CNN | 0.9981 | 0.9981 | 0.9976 |
| AlexNet CNN | 0.9985 | 0.9984 | 0.9983 |
| ResNet-18 CNN | 0.9993 | 0.9993 | 0.9992 |

## 5. Recommended Model

ResNet-18 CNN is the recommended final model from the lecture-aligned comparison because it achieved the highest validation accuracy and matched the highest test accuracy among the new models while remaining clearly aligned with the lecture CNN architecture set.
