#This code is an attempt to dynamically create prompts in order to construct a robust deepfake dataset.
#It will create a list of prompts that will be used for image generation. Prompts will be varied on the following attributes:

import random

#Variable Attributes


#Personal Attributes (Stuff about the subject) 
possible_genders = ["Male", "Female"] #Consider wording Man/Woman or Boy/Girl
possible_ages = ["Young", "Middle-aged", "Old"] #Consider just doing an age range
possible_ethnicities = ["Caucasian", "African American", "East Asian", "Hispanic", "Middle Eastern", "Indian"]
#possible_ethnicities = ["White", "Black", "Asian", "Latino", "Middle Eastern", "Indian"] #More informal terms
#possible_skin_tones = ["Light", "Medium", "Dark"] #Possible alternative or addition to ethnicities
possible_expressions = ["Happy", "Sad", "Angry", "Surprised", "Neutral"] 

#This section should be cognizant of the fact that certain physical characterstics may not correlate well with certain personal attributes. 
#Ex. a bald woman may not be a good image to use for a deepfake dataset.
possible_hair_colors = ["Black", "Brown", "Blonde", "Red", "Gray"]
possible_hair_styles = ["Short", "Medium", "Long", "Bald"] 
possible_heights = ["Short", "Medium", "Tall"] #Consider using a neumerical range of heights or excluding this all together.
possible_body_types = ["Slim", "Athletic", "Average", "Overweight", "Obese"] #Don't really know how to phrase this one nicely

#possible_skin_markings = ["Freckles", "Scars", "Tattoos", "Moles", "Birthmarks"] #Usure about this one

possible_clothing_styles = ["Casual", "Formal", "Business Casual"] #maybe consider specific clothing items
accessories = ["Glasses", "Hat", "Jewelry"] 

#Image Attributes (Stuff about the image)
possible_backgrounds = ["Nature", "Urban", "Indoor", "Outdoor"] #maybe split into categories and use subcategories
possible_background_complexity = ["Simple", "Detailed"] #maybe split into categories and use subcategories
possible_lighting_conditions = ["Day", "Night", "Artificial Light"] #maybe split into categories and use subcategories
possible_depth_of_field = ["Shallow (Blurred Background)", "Deep (Everything in Focus)"] #maybe split into categories and use subcategories
possible_camera_types = ["DSLR", "Mirrorless", "Smartphone", "Webcam", "Surveillance Camera"] #maybe split into categories and use subcategories

#possible_camera_angles = ["Close-up", "Medium Shot", "Wide Shot"] #Don't know if its good to specify this. 
# Looking for shots that have the upper body

image_sizes = [256, 2048] #This is range that will determine the size of the image. Ex. x and y are between images_sizes[0] and images_sizes[1]



#Prompt generation logic.


#Subject attributes
always_present_attributes = []
always_present_attributes.append(possible_genders)
always_present_attributes.append(possible_ages)
always_present_attributes.append(possible_ethnicities)

sometimes_present_attributes = []
sometimes_present_attributes.append(possible_expressions)
sometimes_present_attributes.append(possible_hair_colors)
sometimes_present_attributes.append(possible_hair_styles)
sometimes_present_attributes.append(possible_heights)
sometimes_present_attributes.append(possible_body_types)
sometimes_present_attributes.append(possible_clothing_styles)
sometimes_present_attributes.append(accessories)

for i in range(len(always_present_attributes)):
    always_present_attributes[i] = random.choice(always_present_attributes[i])

for i in range(len(sometimes_present_attributes)):
    if random.random() < 0.25:
        sometimes_present_attributes[i] = random.choice(sometimes_present_attributes[i])
    else:
        sometimes_present_attributes[i] = None
    
subject_attributes = {
    "Gender": always_present_attributes[0],
    "Age": always_present_attributes[1],
    "Ethnicity": always_present_attributes[2],
    "Expression": sometimes_present_attributes[0],
    "Hair Color": sometimes_present_attributes[1],
    "Hair Style": sometimes_present_attributes[2],
    "Height": sometimes_present_attributes[3],
    "Body Type": sometimes_present_attributes[4],
    "Clothing Style": sometimes_present_attributes[5],
    "Accessories": sometimes_present_attributes[6],
}

#Remove any "None" valued keys
subject_attributes = {k: v for k, v in subject_attributes.items() if v is not None}


#Image attributes

image_attributes = {
    "Background": random.choice(possible_backgrounds),
    "Background Complexity": random.choice(possible_background_complexity),
    "Lighting Conditions": random.choice(possible_lighting_conditions),
    "Depth of Field": random.choice(possible_depth_of_field),
    "Camera Type": random.choice(possible_camera_types),
    "Image Size": "{}x{}".format(random.randint(image_sizes[0], image_sizes[1]), random.randint(image_sizes[0], image_sizes[1])),
}

print("Subject Attributes:", subject_attributes)
print("Image Attributes:", image_attributes)

