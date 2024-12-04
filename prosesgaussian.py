import os
import shutil
from PIL import Image, ImageFilter


def clone_directory_structure(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def apply_gaussian_blur(src_dir, dst_dir):
    images_src_dir = os.path.join(src_dir, 'images')
    images_dst_dir = os.path.join(dst_dir, 'images')

    for root, _, files in os.walk(images_src_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, images_src_dir)
                dst_path = os.path.join(images_dst_dir, rel_path)

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                image = Image.open(src_path)
                blurred_image = image.filter(ImageFilter.GaussianBlur(2))  # Adjust the blur radius as needed
                blurred_image.save(dst_path)


def main():
    src_dir = 'test'  # Update with your dataset path (e.g., 'test')
    dst_dir = 'test_gaussian'  # Update with your desired output path

    # Clone the directory structure
    clone_directory_structure(src_dir, dst_dir)

    # Apply Gaussian Blur filter to images
    apply_gaussian_blur(src_dir, dst_dir)


if __name__ == "__main__":
    main()
