import os
import shutil
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np


def clone_directory_structure(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def apply_motion_blur(image):
    size = 15  # Size of the motion blur kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    return cv2.filter2D(np.array(image), -1, kernel_motion_blur)


def apply_low_res(image, scale=0.75):
    width, height = image.size
    low_res = image.resize((int(width * scale), int(height * scale)), Image.BILINEAR)
    return low_res.resize((width, height), Image.NEAREST)


def apply_darkness(image, factor=0.7):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_filters(src_dir, dst_dir, filter_function):
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
                filtered_image = filter_function(image)
                if isinstance(filtered_image, np.ndarray):
                    filtered_image = Image.fromarray(filtered_image)
                filtered_image.save(dst_path)


def main():
    src_dir = 'test'  # Update with your dataset path (e.g., 'test')
    dst_dirs = {
        'test_motion_blur': apply_motion_blur,
        'test_low_res': apply_low_res,
        'test_darkness': apply_darkness,
    }

    for dst_dir, filter_function in dst_dirs.items():
        # Clone the directory structure
        clone_directory_structure(src_dir, dst_dir)

        # Apply the specified filter to images
        apply_filters(src_dir, dst_dir, filter_function)


if __name__ == "__main__":
    main()
