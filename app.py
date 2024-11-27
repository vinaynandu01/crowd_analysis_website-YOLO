from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO('yolov5s.pt')  # Replace with your YOLO model path

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Read the image from the request
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Perform YOLO detection
    results = model.predict(source=img, conf=0.5)  # Confidence threshold
    detections = results[0].boxes.xyxy  # Bounding boxes
    labels = results[0].boxes.cls.cpu().numpy()  # Class labels
    human_boxes = [box for box, label in zip(detections, labels) if int(label) == 0]  # Filter humans

    # Draw bounding boxes for humans only
    human_count = 0
    for box in human_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        human_count += 1

    # Convert the processed image to Base64
    _, buffer = cv2.imencode('.jpg', img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'count': human_count, 'image': encoded_image})

if __name__ == '__main__':
    app.run(debug=True)
