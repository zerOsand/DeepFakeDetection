from transformers import pipeline
import os


class TransformerModel:
    def __init__(self, model_name="prithivMLmods/Deep-Fake-Detector-Model", device=0):
        # Load the model
        # import torch
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = pipeline("image-classification", model=model_name, device=device)
        #underlying_model = self.pipe.model
        # import torch.onnx
        # torch.onnx.export(
        #     underlying_model,
        #     torch.randn(1, 3, 224, 224),
        #     "model.onnx",
        #     export_params=True,
        #     opset_version=18,
        #     do_constant_folding=True,
        #     input_names=["input"],
        #     output_names=["output"],
        # )

    def get_images_from_folder(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                images.append(os.path.join(folder_path, filename))
        return images

    def predict(self, image):
        # Get the prediction
        results = self.pipe(image)
        # Return the results
        return results

    def predict_images(self, images):
        # Get the predictions for all images
        results = self.pipe(images)
        # Return the results
        return results

    def preprocess(self, image):
        # We don't preprocess anything for this model
        return image

    def preprocess_images(self, images):
        # We don't preprocess anything for this model
        return images

    def postprocess(self, prediction_result):
        # Process a single prediction result
        # print("r", prediction_result)
        # Check which label has the highest score
        # -- Model automatically has max_score = prediction_result[0]
        # -- If we notice that later a score below 50% is classified as the 'prediction'
        #   replace the below code with this:
        # prediction = prediction_result[0] if max(prediction_result[0]['score'], prediction_result[1]['score']) else prediction_result[1]
        #print(prediction_result)
        prediction = prediction_result[0]
        prediction["prediction"] = prediction.pop("label")
        prediction["confidence"] = prediction.pop("score")

        # Return the processed result
        return prediction

    def postprocess_images(self, prediction_results):
        # Process all prediction results
        processed_results = []
        for result in prediction_results:
            processed_result = self.postprocess(result)
            processed_results.append(processed_result)
        return processed_results
