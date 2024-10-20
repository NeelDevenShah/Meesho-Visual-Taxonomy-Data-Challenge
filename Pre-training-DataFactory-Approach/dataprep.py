import csv
import json
import os
import re

# TODO: Take care of the length in the inferene submission thing
# TODO: Make the entry of dummy vaible for the cases where there is null values

# Paths
csv_file_path = '/content/Meesho-Data-Challenge/data/train.csv'  # Path to your CSV file
image_folder_path = '/content/Meesho-Data-Challenge/data/train_images'  # Folder where your images are stored
output_json_file = '/content/Meesho-Data-Challenge/data/processed_train.json'  # Output JSON file

# Initialize list to store the dataset
dataset = []

content_dictionary = {
    'Men Tshirts': "give me values for these attributes :: **color** : ['default', 'multicolor', 'black', 'white'], **neck** : ['round', 'polo'], **pattern** : ['printed', 'solid'], **print_or_pattern_type** : ['default', 'solid', 'typography'], **sleeve_length** : ['short sleeves', 'long sleeves']",

    'Sarees': "give me values for these attributes :: **blouse_pattern** : ['same as saree', 'solid', 'same as border', 'default'], **border** : ['woven design', 'zari', 'no border', 'solid', 'default', 'temple border'], **border_width** : ['small border', 'big border', 'no border'], **color** : ['multicolor', 'cream', 'white', 'default', 'navy blue', 'yellow', 'green', 'pink'], **occasion** : ['party', 'traditional', 'daily', 'wedding'], **ornamentation** : ['jacquard', 'default', 'tassels and latkans'], **pallu_details** : ['woven design', 'same as saree', 'default', 'zari woven'], **pattern** : ['zari woven', 'woven design', 'default', 'solid', 'printed'], **print_or_pattern_type** : ['applique', 'elephant', 'floral', 'ethnic motif', 'peacock', 'default', 'solid', 'checked', 'botanical'], **transparency** : ['no', 'yes']",
    
    'Kurtis': "give me values for these attributes :: **color** : ['black', 'red', 'navy blue', 'maroon', 'green', 'pink', 'blue', 'purple', 'grey', 'yellow', 'white', 'multicolor', 'orange'], **fit_shape** : ['straight', 'a-line'], **length** : ['knee length', 'calf length'], **occasion** : ['daily', 'party'], **ornamentation** : ['net', 'default'], **pattern** : ['solid', 'default'], **print_or_pattern_type** : ['solid', 'default'], **sleeve_length** : ['three-quarter sleeves', 'short sleeves', 'sleeveless'], **sleeve_styling** : ['regular', 'sleeveless']",
    
    'Women Tshirts': "give me values for these attributes :: **color** : ['multicolor', 'yellow', 'black', 'default', 'pink', 'maroon', 'white'], **fit_shape** : ['loose', 'boxy', 'regular'], **length** : ['long', 'crop', 'regular'], **pattern** : ['default', 'solid', 'printed'], **print_or_pattern_type** : ['default', 'quirky', 'solid', 'graphic', 'funky print', 'typography'], **sleeve_length** : ['default', 'long sleeves', 'short sleeves'], **sleeve_styling** : ['regular sleeves', 'cuffed sleeves'], **surface_styling** : ['default', 'applique']",
    
    'Women Tops & Tunics': "give me values for these attributes :: **color** : ['black', 'navy blue', 'red', 'default', 'maroon', 'white', 'green', 'blue', 'pink', 'yellow', 'peach', 'multicolor'], **fit_shape** : ['regular', 'fitted', 'boxy', 'default'], **length** : ['crop', 'regular'], **neck_collar** : ['high', 'round neck', 'stylised', 'sweetheart neck', 'v-neck', 'square neck', 'default'], **ocassion** : ['casual', 'party'], **pattern** : ['default', 'printed', 'solid'], **print_or_pattern_type** : ['solid', 'typography', 'graphic', 'default', 'quirky', 'floral'], **sleeve_length** : ['short sleeves', 'sleeveless', 'three-quarter sleeves', 'long sleeves'], **sleeve_styling** : ['regular sleeves', 'default', 'sleeveless', 'puff sleeves'], **surface_styling** : ['knitted', 'default', 'ruffles', 'waist tie-ups', 'tie-ups', 'applique']"
}

inference_dictionary = {
    'Men Tshirts': "**color** : {color}, **neck** : {neck}, **pattern** : {pattern}, **print_or_pattern_type** : {print_or_pattern_type}, **sleeve_length** : {sleeve_length}",
    
    'Sarees': "**blouse_pattern** : {blouse_pattern}, **border** : {border}, **border_width** : {border_width}, **color** : {color}, **occasion** : {occasion}, **ornamentation** : {ornamentation}, **pallu_details** : {pallu_details}, **pattern** : {pattern}, **print_or_pattern_type** : {print_or_pattern_type}, **transparency** : {transparency}",
    
    'Kurtis': "**color** : {color}, **fit_shape** : {fit_shape}, **length** : {length}, **occasion** : {occasion}, **ornamentation** : {ornamentation}, **pattern** : {pattern}, **print_or_pattern_type** : {print_or_pattern_type}, **sleeve_length** : {sleeve_length}, **sleeve_styling** : {sleeve_styling}",
    
    'Women Tshirts': "**color** : {color}, **fit_shape** : {fit_shape}, **length** : {length}, **pattern** : {pattern}, **print_or_pattern_type** : {print_or_pattern_type}, **sleeve_length** : {sleeve_length}, **sleeve_styling** : {sleeve_styling}, **surface_styling** : {surface_styling}",
    
    'Women Tops & Tunics': "**color** : {color}, **fit_shape** : {fit_shape}, **length** : {length}, **neck_collar** : {neck_collar}, **ocassion** : {ocassion}, **pattern** : {pattern}, **print_or_pattern_type** : {print_or_pattern_type}, **sleeve_length** : {sleeve_length}, **sleeve_styling** : {sleeve_styling}, **surface_styling** : {surface_styling}"
}

attr_length_dictionary = {
    'Men Tshirts': 5,
    'Sarees': 10,
    'Kurtis': 9,
    'Women Tshirts': 8,
    'Women Tops & Tunics': 10
}

def start_process():

    # Open and read the CSV file
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            # Extract image filename from the image link
            image_url = row['id']

            image_url = image_url.zfill(6) + '.jpg'

            # Construct the image path in your train folder
            image_path = os.path.join(image_folder_path, image_url)
            # Ensure the image exists in the train/ folder before adding to the dataset
            if os.path.exists(image_path):  

                Category = row['Category']

                content_statement = content_dictionary[Category]
                inference_statement = inference_dictionary[Category]
                attr_length = attr_length_dictionary[Category]

                matches = re.findall(r'\*\*(.*?)\*\*', inference_statement)
                for num, attr in enumerate(matches):
                    inference_statement.replace("{"+attr+"}", row[f'attr_{num+1}'])

                # Construct the conversation in ShareGPT format
                conversation = {
                    "messages": [
                        {
                            "content": content_statement,
                            "role": "user"
                        },
                        {
                            "content": inference_statement,
                            "role": "assistant"
                        }
                    ],
                    "images": [
                        image_path
                    ]
                }

                # Add the conversation to the dataset
                dataset.append(conversation)
            else:
                print(f"Image {image_url} not found in {image_folder_path}, skipping.")

    # Save the dataset to a JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

    print(f"Dataset successfully saved to {output_json_file}")

if __name__ == "__main__":
  start_process()