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

real = "sample_input/real"
fake = "sample_input/fake"

# Predict on an image

def predict_on_images(images):
    for image in images:
        result = pipe(image)
        print(result)

real_images = get_images_from_folder(real)
fake_images = get_images_from_folder(fake)  
# print(real_images)
# print(fake_images)

print("Real Images")
predict_on_images(real_images)

print("Fake Images")
predict_on_images(fake_images)