
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Step 1: Load YOLOv5 Model
print("Loading YOLOv5 model...")
model = YOLO("yolov5su.pt")

# Step 2: Load Image
image_path = r"E:\tktgroup-white-flowers_777078-40685.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Image not found! Check image path: {image_path}"
    )

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 3: Perform Object Detection
print("Performing object detection...")

results = model.predict(
    source=image_rgb,
    conf=0.25,
    save=False
)

# Step 4: Display Results
annotated_image = results[0].plot()

annotated_rgb = cv2.cvtColor(
    annotated_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(12, 8))
plt.imshow(annotated_rgb)
plt.axis("off")
plt.title("Object Detection Results")
plt.show()

# Step 5: Save Output Image
output_path = "output.jpg"

cv2.imwrite(output_path, annotated_image)

print("Image saved as:", output_path)

# Step 6: Print Detected Objects
print("\nDetected Objects:")

for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = box.conf[0].item()
    cls = int(box.cls[0].item())

    print(
        f"Class: {results[0].names[cls]}, "
        f"Confidence: {conf:.2f}, "
        f"Coordinates: ({x1:.2f}, {y1:.2f}), "
        f"({x2:.2f}, {y2:.2f})"
    )
