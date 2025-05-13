from pathlib import Path
from PIL import Image
import onnxruntime as ort
import numpy as np
import process.facedetector as facedetector  # Ensure this import exists or add it

class TransformerModelDimaONNX:
    def __init__(self, model_path="onnx_models/dima_transformer.onnx", resolution=224):
        # Convert model_path to a Path object
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),  # Convert Path object to string for onnxruntime
        )
        self.resolution = resolution
        self.valid_extensions = (".jpg", ".jpeg", ".png")
        print("Initialized TransformerModelDimaONNX")

    #     self.inspect_model_inputs()  # Inspect model inputs during initialization

    # def inspect_model_inputs(self):
    #     # Print the input details of the ONNX model
    #     print("Model Input Details:")
    #     for input_meta in self.session.get_inputs():
    #         print(f"Name: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")

    def apply_transforms(self, image):
        # Resize the image to the required resolution
        image = image.resize((self.resolution, self.resolution))
        # Convert the image to a numpy array
        image = np.array(image).astype(np.float32)
        # Normalize the image
        image = image / 255.0
        # Reorder dimensions to (channels, height, width)
        image = np.transpose(
            image, (2, 0, 1)
        )  # Convert from (height, width, channels) to (channels, height, width)
        # Add a batch dimension
        image = np.expand_dims(
            image, axis=0
        )  # Shape becomes (batch_size, channels, height, width)
        return image

    def predict(self, image):
        # Get the prediction using the correct input key
        results = self.session.run(
            None, {"pixel_values": image}
        )  # Use "pixel_values" as the input key
        # Return the results
        return results[0]

    def preprocess(self, image, facecrop=None):
        if facecrop:
            self.resolution_ratio = 1.5  # Default value if not set
            # Assuming faceDetector returns (center_x, center_y) or None
            # The faceDetector might need a specific input format (e.g., numpy array)
            face_center_coords = None
            try:
                # Convert PIL Image to numpy array (RGB)
                np_image = np.array(image.convert('RGB'))

                # Assuming faceDetector takes a numpy array and the ONNX session
                detectorResults = facedetector.faceDetector(np_image, face_detector=facecrop)

                face_center_coords = detectorResults[3]  # Get the center coordinates of the detected face
                already_headshot = detectorResults[4]  # Check if the image is already a headshot
            except Exception as e:
                # Handle potential errors during face detection
                print(f"Warning: Face detection failed with error: {e}. Proceeding without cropping.")

            if already_headshot:
                #print("Image is already a headshot. No cropping needed.")
                return self.apply_transforms(image)

            if face_center_coords is not None:
                # Define the ratio for cropping relative to the base resolution.
                if not hasattr(self, 'resolution_ratio'):
                    print("Warning: self.resolution_ratio not set. Defaulting to 1.0.")
                    self.resolution_ratio = 1.0 # Default value if not set

                center_x, center_y = face_center_coords
                img_width, img_height = image.size
                # Desired crop size based on resolution and ratio, centered around the face
                crop_half_size = int(self.resolution * self.resolution_ratio / 2) # Half the desired dimension

                # Calculate initial crop box coordinates
                left = center_x - crop_half_size
                top = center_y - crop_half_size
                right = center_x + crop_half_size
                bottom = center_y + crop_half_size

                # Clamp coordinates to be within image boundaries
                left = max(0, left)
                top = max(0, top)
                right = min(img_width, right)
                bottom = min(img_height, bottom)

                # Ensure coordinates are integers for PIL crop
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)

                # Check if the calculated box has valid dimensions (width > 0 and height > 0)
                if right > left and bottom > top:
                    # Crop the original PIL image
                    image = image.crop((left, top, right, bottom))
                    # Optional: Save the cropped image for debugging
                    # Path("temp").mkdir(exist_ok=True) # Ensure temp directory exists
                    # image.save(f"temp/cropped_{center_x}_{center_y}.jpg")
                else:
                    # Optional: Log or print a warning if the crop box is invalid
                    print(f"Warning: Invalid crop box calculated ({left}, {top}, {right}, {bottom}) for face at ({center_x}, {center_y}). Using original image.")

        # Apply standard transforms to the (potentially cropped) image
        return self.apply_transforms(image)

    def preprocess_images(self, images):
        # We don't preprocess anything for this model
        # Resize the images to the required resolution
        for i in range(len(images)):
            images[i] = self.apply_transforms(images[i])
        return images

    def postprocess(self, prediction_result):
        # Process a single prediction result
        # print("r", prediction_result)
        # Check which label has the highest score
        # -- Model automatically has max_score = prediction_result[0]
        # -- If we notice that later a score below 50% is classified as the 'prediction'
        #   replace the below code with this:
        # prediction = prediction_result[0] if max(prediction_result[0]['score'], prediction_result[1]['score']) else prediction_result[1]
        # print("--" * 20)
        # print("Prediction result:", prediction_result)
        # Apply softmax to normalize the prediction result
        exp_scores = np.exp(prediction_result[0])  # Exponentiate the scores
        probabilities = exp_scores / np.sum(exp_scores)
        # Normalize by dividing by the sum of exponentiated scores

        confidence = float(max(probabilities))
        raw_label = "real" if probabilities[0] > probabilities[1] else "fake"
        strength = (
            "likely"
            if confidence < 0.2 or confidence > 0.8
            else "weakly" if confidence < 0.4 or confidence > 0.6 else "uncertain"
        )

        if strength == "uncertain":
            label = "uncertain"
        else:
            label = f"{strength} {raw_label}"

        prediction = {"prediction": label, "confidence": confidence}
        # Return the processed result
        return prediction

    def postprocess_images(self, prediction_results):
        # Process all prediction results
        processed_results = []
        for result in prediction_results:
            processed_result = self.postprocess(result)
            processed_results.append(processed_result)
        return processed_results
