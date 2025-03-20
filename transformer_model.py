from transformers import pipeline

# Load the model
pipe = pipeline('image-classification', model="prithivMLmods/Deep-Fake-Detector-Model", device=0)

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
        result = pipe(image) #List of dictionaries in the form [{'label': 'Fake', 'score': 0.9999}, {'label': 'Real', 'score': 0.0001}]
        #add the image path to the last element of the list
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
        #check which label has the highest score
        prediction = max(result, key=lambda x: x['score'])
        image_path = prediction.pop('image_path', None)  # Remove image path from the result
        # Add image path back to the prediction
        prediction['image_path'] = image_path
        # Add the processed result to the list
        processed_results.append((image_path, prediction))
    return processed_results


#source CS564/bin/activate