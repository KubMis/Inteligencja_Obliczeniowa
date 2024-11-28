from ultralytics import YOLO

model = YOLO('yolov8n.pt')
input_path = 'images'

results = model.predict(source=input_path,save=True,project='output')

for result in results:
    for box in result.boxes.data:
        x1, y1, x2, y2, confidence, cls = box.tolist()
        print(f'Object: {model.names[int(cls)]}, confidence: {confidence:.2f}')
