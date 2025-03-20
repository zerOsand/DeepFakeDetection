from transformers import pipeline

# Load the model
pipe = pipeline(
    "image-classification", model="prithivMLmods/Deep-Fake-Detector-Model", device=0
)


def get_images_from_folder(folder_path):
    import os

    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            images.append(os.path.join(folder_path, filename))
    return images


def predict(images):
    results = []
    for image in images:
        result = pipe(
            image
        )  # List of dictionaries in the form [{'label': 'Fake', 'score': 0.9999}, {'label': 'Real', 'score': 0.0001}]
        # add the image path to the last element of the list
        result.append(image)
        results.append(result)
    return results


def preprocess(anything):
    # Placeholder for preprocessing logic
    return anything


def postprocess(prediction_results):
    # Return a list of tuples with image path and prediction
    processed_results = []
    for result in prediction_results:
        print("r", result)
        # Check which label has the highest score
        # -- Model automatically has max_score = result[0]
        # -- If we notice that later a score below 50% is classified as the 'prediction'
        #   replace the below code with this:
        # prediction = result[0] if max(result[0]['score'], result[1]['score']) else result[1]
        prediction = result[0]
        # Add image path back to the prediction
        prediction["image_path"] = result[2]
        # Add the processed result to the list
        processed_results.append(prediction)
    return processed_results


# source CS564/bin/activate
