import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO, SAM

BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_MODEL_PATH = BASE_DIR / 'media/models/best.pt'
SAM_WEIGHTS_PATH = BASE_DIR / 'media/models/sam2_b.pt'

_yolo_model = None
_sam_model = None

def load_models(device='cpu'):
    global _yolo_model, _sam_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
    if _sam_model is None:
        _sam_model = SAM(str(SAM_WEIGHTS_PATH))
    return _yolo_model, _sam_model

def run_detection_and_segmentation(image_path, output_dir, device='cpu', conf=0.3):
    os.makedirs(output_dir, exist_ok=True)

    yolo_model, sam_model = load_models(device=device)

    # Run YOLO detection
    results = yolo_model(str(image_path), device=device, conf=conf)
    result = results[0]

    # Read original image
    orig_img = cv2.imread(str(image_path))
    h, w = orig_img.shape[:2]

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    detection_img = orig_img.copy()
    tumor_summary = []

    if hasattr(result, 'boxes') and len(result.boxes):
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_names = result.names

        for i, bbox in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = bbox.astype(int)
            cls_id = int(classes[i])
            conf = confidences[i]
            label = f"{class_names[cls_id]} {conf*100:.1f}%"
            tumor_summary.append(f"{class_names[cls_id]} ({conf*100:.1f}%)")

            # Draw detection bounding box
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                detection_img, label, (x1, max(0, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
            )

            # SAM segmentation
            sam_results = sam_model(
                orig_img,
                bboxes=np.array([[x1, y1, x2, y2]]),
                verbose=False,
                device=device
            )

            try:
                mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
            except Exception:
                mask = sam_results[0].masks[0]

            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)

    # Blue segmentation overlay with higher opacity
    overlay_img = orig_img.copy()
    overlay_color = np.zeros_like(orig_img)
    overlay_color[:, :, 0] = 255  # Blue channel

    alpha = 0.9  # opacity (0.0 - transparent, 1.0 - fully opaque)
    overlay_img = cv2.addWeighted(overlay_img, 1.0, overlay_color * combined_mask[:, :, None], alpha, 0)



    # Save images
    base_name = Path(image_path).stem
    detection_filename = f"{base_name}_detection.png"
    segmentation_filename = f"{base_name}_segmentation.png"

    cv2.imwrite(str(Path(output_dir) / detection_filename), detection_img)
    cv2.imwrite(str(Path(output_dir) / segmentation_filename), overlay_img)

    return detection_filename, segmentation_filename, tumor_summary
