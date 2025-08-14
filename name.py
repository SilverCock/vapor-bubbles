import os
import json

image_dir = 'G:\\experement\\35_t\\dataset\\raw'  # замени на нужный путь
output_dir = image_dir  # можно изменить

for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        name, _ = os.path.splitext(filename)  # разделение имени и расширения
        json_filename = name + '.json'
        json_path = os.path.join(output_dir, json_filename)

        if not os.path.exists(json_path):
            empty_annotation = {
                "version": "5.0.1",
                "flags": {},
                "shapes": [],
                "imagePath": filename,
                "imageData": None,
                "imageHeight": 200,  # укажи реальные значения
                "imageWidth": 300
            }
            with open(json_path, 'w') as f:
                json.dump(empty_annotation, f, indent=4)
