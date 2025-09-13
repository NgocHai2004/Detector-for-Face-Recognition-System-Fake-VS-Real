import cv2
import torch
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image


# Load YOLOv8 (face detection)
yolo_model = YOLO("detect/yolov8n.pt")  # bạn cần file weight detect mặt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = clf_model.fc.in_features
clf_model.fc = torch.nn.Linear(num_features, 2)
clf_model.load_state_dict(torch.load("resnet18_fakeface.pth", map_location=device))
clf_model = clf_model.to(device)
clf_model.eval()


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = ["fake", "real"]


cap = cv2.VideoCapture(0)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # B1: YOLO detect face
    results = yolo_model(frame, conf=0.5)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)  # to numpy
        for box in boxes:
            x1, y1, x2, y2 = box
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # B2: Transform face
            pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            input_tensor = test_transform(pil_img).unsqueeze(0).to(device)

            # B3: Predict real/fake
            with torch.no_grad():
                output = clf_model(input_tensor)
                _, pred = torch.max(output, 1)
                pred_class = class_names[pred.item()]

            # Vẽ bounding box + label
            color = (0, 255, 0) if pred_class == "real" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, pred_class, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("YOLOv8 + FakeFace Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
