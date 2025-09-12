import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
# print(model.names)
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if not ret:
        break

    result = model(frame)
    for r in result:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"Face {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    cv2.imshow("YOLOv8 Face Detection", frame)

    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()