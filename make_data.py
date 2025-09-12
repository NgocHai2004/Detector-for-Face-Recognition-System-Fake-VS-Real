import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class MyDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = []
        self.labels = []
        self.categories = ["fake", "real"]   
        self.transform = transform

        if train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")

        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for item in os.listdir(data_files):
                path = os.path.join(data_files, item)
                self.image_paths.append(path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================
# Ví dụ dùng:
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = MyDataset("dataset_split", train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# images, labels = next(iter(train_loader))
# print(images.shape)  
# print(labels)      
images, labels = next(iter(train_loader))


img = images[0]  
label = labels[0].item()

# Chuyển tensor -> numpy để hiển thị
img = img.permute(1, 2, 0).numpy()  # từ [C,H,W] -> [H,W,C]

# Undo normalize (-1..1 -> 0..1)
img = (img * 0.5) + 0.5  

plt.imshow(img)
plt.title(f"Label: {label}")  # 0 = fake, 1 = real
plt.axis("off")
plt.show()

