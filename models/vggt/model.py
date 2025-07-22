# model.py
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class Model:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image):
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()