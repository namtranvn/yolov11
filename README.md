git remote remove origin
git remote add origin https://@github.com/namtranvn/yolov11.git

git add .
git commit -m "first commit"
git push -u origin

git push --set-upstream origin main


import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
model_path = "model.onnx"
session = ort.InferenceSession(model_path)

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
_, _, height, width = input_shape  # Assuming NCHW format

# Open video
video_path = "video.mp4"  # Use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Load class names
class_names = []
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    resized = cv2.resize(frame, (width, height))
    blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = blob.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  # HWC to CHW
    blob = np.expand_dims(blob, axis=0)  # Add batch dimension
    
    # Run inference
    outputs = session.run(None, {input_name: blob})
    
    # Post-process (assuming YOLOv5-like output format)
    predictions = outputs[0][0]  # First batch
    
    # Filter by confidence
    mask = predictions[:, 4] > 0.5  # Confidence threshold
    filtered_preds = predictions[mask]
    
    if len(filtered_preds) > 0:
        # Get class scores and IDs
        class_scores = filtered_preds[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        
        # Get boxes
        boxes = filtered_preds[:, :4]
        
        # Convert to corner coordinates and scale to original image
        orig_h, orig_w = frame.shape[:2]
        
        for i, box in enumerate(boxes):
            # Convert center_x, center_y, w, h to x1, y1, x2, y2
            x, y, w, h = box
            x1 = int((x - w/2) / width * orig_w)
            y1 = int((y - h/2) / height * orig_h)
            x2 = int((x + w/2) / width * orig_w)
            y2 = int((y + h/2) / height * orig_h)
            
            # Get class info
            class_id = class_ids[i]
            confidence = class_scores[i, class_id] * filtered_preds[i, 4]
            
            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            if class_id < len(class_names):
                label = f"{class_names[class_id]}: {confidence:.2f}"
            else:
                label = f"Class {class_id}: {confidence:.2f}"
                
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

