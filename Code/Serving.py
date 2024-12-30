import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import random
import warnings
import os
warnings.filterwarnings(action='ignore')

os.makedirs("imgs", exist_ok=True)
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates") # 템플릿 파일이 있는 디렉토리 지정
app = FastAPI()

# Number of classes
num_classes = 10
# CIFAR-10 클래스 이름
CIFAR10_CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

dataset_path = './data/cifar10'

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 입력 크기에 맞춤
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                           download=True, transform=transform_test)

# Initialize and modify ResNet50 model for CIFAR-10
def initialize_model(num_classes=10):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(model.fc.in_features),  # Batch Normalization 추가
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = initialize_model(num_classes=num_classes)
model.load_state_dict(torch.load('fine_tuned_resnet50_cifar10_best_5.pth', map_location=device))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

def test_image():
    idx = random.randint(1, len(testset))
    print("idx=", idx)
    label = testset[idx][1]
    img = testset[idx][0].unsqueeze(0).to(device)  # Move the input image tensor to the GPU
    model.eval()

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)

    torchvision.utils.save_image(img.cpu(), "./imgs/f.jpg")

    return predicted.item(), label

    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {classes[predicted]}, True: {classes[label]}')
    plt.show()

test_image()


@app.get("/test")
async def read_root(request: Request):
    predicted, label = test_image()
    result = f'Predicted: {CIFAR10_CLASSES[predicted]}  True: {CIFAR10_CLASSES[label]}'
    context = {"name": result}
    return templates.TemplateResponse("index1030.html", {"request": request, "context": context})

app.mount("/imgs", StaticFiles(directory="imgs"), name='images')