import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ======================
# Load lại model đã train
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # 2 classes
model.load_state_dict(torch.load("resnet18_fakeface.pth", map_location=device))
model = model.to(device)
model.eval()

# ======================
# Transform test (giống khi train)
# ======================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================
# Dự đoán cho 1 ảnh bất kỳ
# ======================
img_path = "test.jpg"   # đổi đường dẫn ảnh của bạn
image = Image.open(img_path).convert("RGB")
input_tensor = test_transform(image).unsqueeze(0).to(device)

# Forward
with torch.no_grad():
    output = model(input_tensor)
    _, pred = torch.max(output, 1)

class_names = ["fake", "real"]  # đổi đúng theo dataset của bạn
pred_class = class_names[pred.item()]

# ======================
# Hiển thị kết quả
# ======================
plt.imshow(image)
plt.title(f"Predicted: {pred_class}")
plt.axis("off")
plt.show()

print("Dự đoán:", pred_class)
