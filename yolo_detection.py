from ultralytics import YOLO
import cv2
import os


def crop_with_expand(model_path, image_path, save_dir="crops", scale=1.2):
    os.makedirs(save_dir, exist_ok=True)

    model = YOLO(model_path)

    results = model(image_path)
    r = results[0]
    img = r.orig_img
    h, w = img.shape[:2]
    boxes = r.boxes

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        conf = boxes.conf[i].item()
        cls_id = int(boxes.cls[i])
        label = r.names[cls_id]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        bw = (x2 - x1) * scale
        bh = (y2 - y1) * scale
        new_x1 = cx - bw / 2
        new_y1 = cy - bh / 2
        new_x2 = cx + bw / 2
        new_y2 = cy + bh / 2
        new_x1 = max(0, int(new_x1))
        new_y1 = max(0, int(new_y1))
        new_x2 = min(w, int(new_x2))
        new_y2 = min(h, int(new_y2))
        crop = img[new_y1:new_y2, new_x1:new_x2]
        save_path = os.path.join(
            save_dir, f"{label}_{i}_conf{conf:.2f}.jpg"
        )
        cv2.imwrite(save_path, crop)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    crop_with_expand(
        model_path="/home/user/YOLO_train/ultralytics-main/runs/detect/herb-yolov5su-100-16/weights/best.pt",
        image_path="/home/user/YOLO_train/ultralytics-main/6_3.jpg",
        scale=1.2
    )
