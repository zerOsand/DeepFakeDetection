import cv2
import numpy as np
import onnxruntime as ort
import os

def faceDetector(orig_image, threshold=0.7, face_detector=None):
    if face_detector is None:
        raise ValueError("face_detector must be provided")

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict_face(
        orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold
    )

    center = None
    if boxes.size > 0:
        first_box = boxes[0]
        # Calculate center coordinates (x, y) for the first box
        center_x = (first_box[0] + first_box[2]) // 2
        center_y = (first_box[1] + first_box[3]) // 2
        center = (center_x, center_y)


    return boxes, labels, probs, center

def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict_face(
    width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1
):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    # print(boxes)
    # print(confidences)

    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        # print(confidences.shape[1])
        probs = confidences[:, class_index]
        # print(probs)
        mask = probs > prob_threshold
        probs = probs[mask]

        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        # print(subset_boxes)
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(
            box_probs,
            iou_threshold=iou_threshold,
            top_k=top_k,
        )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return (
        picked_box_probs[:, :4].astype(np.int32),
        np.array(picked_labels),
        picked_box_probs[:, 4],
    )

# face_detector = ort.InferenceSession(
#     "onnx_models/face_detector.onnx",
#     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
# )

# tempset = "Datasets/deepfake_v4"
# output_dir = "Datasets/deepfake_v4_detections"
# output_boxed_dir = "Datasets/deepfake_v4_boxed" # Directory for images with boxes

# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(output_boxed_dir, exist_ok=True) # Create the new directory

# if __name__ == "__main__":
#     print(f"Processing images in: {tempset}")
#     print(f"Saving detections to: {output_dir}")
#     print(f"Saving boxed images to: {output_boxed_dir}") # Print new dir info

#     for filename in os.listdir(tempset):
#         filepath = os.path.join(tempset, filename)
#         if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             print(f"Processing {filename}...")
#             try:
#                 orig_image = cv2.imread(filepath)
#                 if orig_image is None:
#                     print(f"Warning: Could not read image {filename}. Skipping.")
#                     continue

#                 boxes, labels, probs = faceDetector(orig_image)

#                 # --- Save detection results (existing code) ---
#                 base_filename = os.path.splitext(filename)[0]
#                 output_filename = os.path.join(output_dir, f"{base_filename}_detections.txt")

#                 with open(output_filename, 'w') as f:
#                     if boxes.size > 0:
#                         for i in range(len(boxes)):
#                             box = boxes[i]
#                             label = labels[i]
#                             prob = probs[i]
#                             f.write(f"{label} {prob:.4f} {box[0]} {box[1]} {box[2]} {box[3]}\n")
#                     else:
#                         f.write("No faces detected.\n")
#                 print(f"Saved detections for {filename} to {output_filename}")

#                 # --- Draw boxes and save the image (new code) ---
#                 boxed_image = orig_image.copy() # Work on a copy
#                 if boxes.size > 0:
#                     for i in range(len(boxes)):
#                         box = boxes[i]
#                         prob = probs[i]
#                         # Draw rectangle (BGR color: Green)
#                         cv2.rectangle(boxed_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#                         # Put probability text above the box
#                         label_text = f"{prob:.2f}"
#                         cv2.putText(boxed_image, label_text, (box[0], box[1] - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                 # Save the image with boxes
#                 output_boxed_filename = os.path.join(output_boxed_dir, f"{base_filename}_boxed.jpg")
#                 cv2.imwrite(output_boxed_filename, boxed_image)
#                 print(f"Saved boxed image for {filename} to {output_boxed_filename}")
#                 # --- End of new code ---

#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")

#     print("Processing complete.")

