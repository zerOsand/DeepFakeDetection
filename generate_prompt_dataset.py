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

