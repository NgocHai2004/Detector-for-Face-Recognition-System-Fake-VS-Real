import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Make_data import MyDataset

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),       # lật ngang
    transforms.RandomRotation(10),                # xoay ±10 độ
    transforms.ColorJitter(brightness=0.2, 
                           contrast=0.2, 
                           saturation=0.2, 
                           hue=0.1),              # thay đổi màu
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # crop ngẫu nhiên
    transforms.RandomGrayscale(p=0.1),            # ngẫu nhiên về ảnh xám
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)), # làm mờ nhẹ

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

train_dataset = MyDataset("dataset_split", train=True, transform=train_transform)
val_dataset = MyDataset("dataset_split", train=True, transform=train_transform)
test_dataset = MyDataset("dataset_split", train=True, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) 
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")
    return model
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total
# Đánh giá toàn bộ test set
test_acc = evaluate(model, test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")

# ======================
# 6. Train model
# ======================
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# ======================
# 7. Save model
# ======================
torch.save(model.state_dict(), "resnet18_fakeface.pth")


