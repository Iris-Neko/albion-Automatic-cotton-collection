from PIL import Image
import os

def flip_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_img.save(image_path)

folder_path = "maps"
flip_images_in_folder(folder_path)
