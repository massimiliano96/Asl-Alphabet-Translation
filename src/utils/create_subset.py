import os
import random
import shutil


def copy_images(source_folder, destination_folder, num_images=20):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for root, dirs, files in os.walk(source_folder):
        for directory in dirs:
            source_path = os.path.join(root, directory)
            destination_path = destination_folder

            # Select only files with ".jpg" or ".png" extensions
            image_files = [
                f
                for f in os.listdir(source_path)
                if f.endswith(".jpg") or f.endswith(".png")
            ]

            if len(image_files) < num_images:
                print(f"Warning: Insufficient images in {source_path}. Skipping.")
            else:
                # Randomly select num_images from available images
                selected_images = random.sample(image_files, num_images)
                for image in selected_images:
                    source_image_path = os.path.join(source_path, image)
                    destination_image_path = os.path.join(destination_path, image)
                    shutil.copy2(source_image_path, destination_image_path)


if __name__ == "__main__":
    # Define source and destination directories
    source_directory = "./data/asl_alphabet_train/asl_alphabet_train"
    destination_directory = "data_first_draft"
    num_images_per_folder = 20

    # Copy images from source to destination
    copy_images(
        source_directory, destination_directory, num_images=num_images_per_folder
    )

    print("Script executed successfully.")
