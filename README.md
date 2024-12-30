# DL_CNN_classification_Serving

## 1. Development Environment Assign
- Google Colaboratory
- python 3.10.12

## 2. Data
- ImageNet
- Fine Tuning : Cifar-10

## 3. Model
- ResNet50

## 4. Fine Tuning
1. Batch Normalization
2. Gradient clipping
3. batch size = 32
4. optimizer adam
   - weight decay = 0.001
   - learning rate = 0.0001

## 5. Modeling Result
![image](https://github.com/user-attachments/assets/3d4ae3d7-67ee-458e-898d-11e55b912a69)
accuracy : 81.90%  
Predicted : dog  
True : dog

## 6. Serving
1. State dict
  - fine_tuned_resnet50_cifar10_best_5.pth
2. Resize
  - (224, 224)
3. Batch Normalization

## 7. Web Serving Result
![image](https://github.com/user-attachments/assets/48f88964-ef7b-40f2-a113-ecceb562b4e6)
