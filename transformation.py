import os
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

def save_plant(image, output_path, image_name):
    file_path = os.path.join(output_path, image_name)
    pcv.print_image(image, file_path)
    return file_path

def draw_pcv_circles(image, points, color, thickness=8):
    if points is not None and len(points) > 0:
        for point in points:
            x, y = int(point[0][0]), int(point[0][1])
            cv2.circle(image, (x, y), thickness, color, -1) 
    return image

def process_original(image, image_name, output_path):
    return save_plant(image, output_path, f"{image_name}_original.jpg")

def process_guassian(image, image_name, output_path):
    gray_image = pcv.rgb2gray(rgb_img=image)
    guassian_binary = pcv.threshold.binary(gray_image, threshold=120, object_type='dark')
    return save_plant(guassian_binary, output_path, f"{image_name}_gaussian-binary.jpg"), guassian_binary

def process_damaged_mask(image, image_name, guassian_binary, output_path):
    damaged_mask = np.where(guassian_binary[:, :, None] == 255, image, np.full_like(image, 255))
    return save_plant(damaged_mask, output_path, f"{image_name}_mask.jpg")

def process_roi_object(image, guassian_binary_image, image_name, output_path):
    roi_image = image.copy()
    roi_image[np.where(guassian_binary_image == 255)] = [0, 255, 0]
    x, y, w, h = cv2.boundingRect(guassian_binary_image)
    cv2.rectangle(roi_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
    return save_plant(roi_image, output_path, f"{image_name}_roi_objects.jpg")

def process_pseudolandmarks(image, guassian_binary_image, image_name, output_path):
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=image, mask=guassian_binary_image)
    pseudolandmarks_img = np.copy(image)
    pseudolandmarks_img = draw_pcv_circles(pseudolandmarks_img, top, (255, 0, 0), thickness=10)  
    pseudolandmarks_img = draw_pcv_circles(pseudolandmarks_img, center_v, (0, 0, 255), thickness=10)
    pseudolandmarks_img = draw_pcv_circles(pseudolandmarks_img, bottom, (255, 0, 255), thickness=10) 

    return save_plant(pseudolandmarks_img, output_path, f"{image_name}_pseudolandmarks.jpg")

def process_skeletonize(guassian_binary_image, image_name, output_path):
    skel = pcv.morphology.skeletonize(mask=guassian_binary_image)
    return save_plant(skel, output_path, f"{image_name}_skeleton.jpg")

def process_image(image_path, output_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    image_output_path = os.path.join(output_path)
    os.makedirs(image_output_path, exist_ok=True)

    image, _, _ = pcv.readimage(image_path, mode="native")
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return []

    original_path = process_original(image, image_name, image_output_path)
    guassian_path, guassian_binary_image = process_guassian(image, image_name, image_output_path)
    damaged_mask_path = process_damaged_mask(image, image_name, guassian_binary_image, image_output_path)
    roi_object_path = process_roi_object(image, guassian_binary_image, image_name, image_output_path)
    pseudolandmarks_path = process_pseudolandmarks(image, guassian_binary_image, image_name, image_output_path)
    skeleton_path = process_skeletonize(guassian_binary_image, image_name, image_output_path)

    return [original_path, guassian_path, damaged_mask_path, roi_object_path, pseudolandmarks_path, skeleton_path]

def apply_transformations(input_path, output_path='transformed_images'):
    images = []
    if os.path.isfile(input_path):
        os.makedirs(output_path, exist_ok=True)
        images.append(process_image(input_path, output_path))
        return images
    elif os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for file_name in tqdm(os.listdir(input_path)):
            file_path = os.path.join(input_path, file_name)
            if os.path.isfile(file_path):
                images.append(process_image(file_path, output_path))
        return images
    else:
        print(f"Указанный путь {input_path} не является файлом или директорией.")
        return None

def plot_images(images):
    if not images or not isinstance(images[0], list):
        print("Нет изображений для отображения или неверный формат данных.")
        return

    num_files = len(images)
    total_pages = num_files * 2
    titles = ["Original", "Gaussian blur", "Mask", "ROI Objects", "Pseudolandmarks", "Skeleton"]
    current_index = [0]

    def update_plot():
        plt.close('all') 

        if current_index[0] % 2 == 0:
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            fig.suptitle(f"Объект {current_index[0] // 2 + 1} / {num_files}", fontsize=14, fontweight="bold")

            img_set = images[current_index[0] // 2]
            for i, ax in enumerate(axes.flatten()):
                ax.clear()
                img_path = img_set[i]
                img = cv2.imread(img_path)

                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "Ошибка загрузки", ha="center", va="center", fontsize=12)

                ax.set_title(titles[i], fontsize=10, fontweight="bold")
                ax.axis("off")

            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show(block=True)

        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f"Объект {current_index[0] // 2 + 1}", fontsize=14, fontweight="bold")

            img_path = images[current_index[0] // 2][0]
            img = cv2.imread(img_path)

            if img is not None:
                plot_analyze_histogram(img, ax)

            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show(block=True)

    def on_key(event):
        if event.key == "right" and current_index[0] < total_pages - 1:
            current_index[0] += 1
            update_plot()
        elif event.key == "left" and current_index[0] > 0:
            current_index[0] -= 1
            update_plot()

    update_plot()

def plot_analyze_histogram(image, ax):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    channels = {
        "red": (image[:, :, 0], "red"),
        "green": (image[:, :, 1], "green"),
        "blue": (image[:, :, 2], "blue"),
        "hue": (cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 0], "purple"),
        "saturation": (cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1], "cyan"),
        "value": (cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 2], "orange"),
    }

    ax.clear()
    for label, (channel, color) in channels.items():
        hist, bins = np.histogram(channel, bins=256, range=(0, 256), density=True)
        ax.plot(bins[:-1], hist * 100, color=color, label=label)

    ax.set_title("Color Distribution")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformation')
    parser.add_argument('source', type=Path, help='path to source image or plants dataset')
    parser.add_argument('dest', type=Path, nargs='?', help='path to store transformed images')
    args = parser.parse_args()

    if args.source.is_file():
        plot_images(process_image(args.source, args.dest))
    elif args.source.is_dir():
        if args.dest is not None:
            apply_transformations(args.source, args.dest)
        else:
            raise argparse.ArgumentError(message='No destination path')