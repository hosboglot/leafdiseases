import os
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import cv2

from tqdm import tqdm


def original(image: cv2.Mat):
    return image

def rotate(image: cv2.Mat, angle: int | None = None):
    if angle is None:
        angle = np.random.randint(0, 360)
    size_reverse = np.array(image.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(image, M, tuple(size_new.astype(int)))

def blur(image: cv2.Mat, size: int | None = None):
    if size is None:
        size = np.random.randint(1, 10)
    return cv2.blur(image, (size, size))

def contrast(image: cv2.Mat, factor: float | None = None):
    if factor is None:
        factor = np.random.uniform(1.1, 2)
    return cv2.convertScaleAbs(image, alpha=factor)

def scale(image: cv2.Mat, factor: float | None = None):
    if factor is None:
        factor = np.random.uniform(1.2, 2)
    size = image.shape[:2]
    resized = cv2.resize(image, None, fx=factor, fy=factor)
    new_size = resized.shape[:2]
    return resized[
        (new_size[0] - size[0]) // 2:-(new_size[0] - size[0]) // 2,
        (new_size[1] - size[1]) // 2:-(new_size[1] - size[1]) // 2
    ]

def illuminate(image: cv2.Mat, factor: float | None = None):
    if factor is None:
        factor = np.random.uniform(10, 50)
    return cv2.convertScaleAbs(image, beta=factor)

def distort(image: cv2.Mat, factor: float | None = None):
    if factor is None:
        factor = np.random.uniform(1, 10)
    return (image + np.round(np.random.normal(0, factor, image.shape))).astype(np.uint8)


def augment_dataset(input_path: Path, output_path: Path, target_count: int | None = None, random_seed=7):
    np.random.seed(random_seed)
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    methods = [rotate, blur, contrast, scale, illuminate, distort]

    if target_count is None:
        target_count = max([len(list(category_path.glob('*'))) for category_path in input_path.iterdir()])

    for category_path in input_path.iterdir():
        current_out_path = output_path / os.path.join(*category_path.parts[-1:])
        os.makedirs(current_out_path, exist_ok=True)

        image_names = [f.name for f in category_path.iterdir() if f.suffix.lower() in image_extensions]

        # augment images
        if len(image_names) < target_count:
            for _ in tqdm(range(target_count - len(image_names))):
                image_name = Path(np.random.choice(image_names))
                method = np.random.choice(methods)
                augmented_image_path = current_out_path / (image_name.with_stem(image_name.stem + '_' + method.__name__))

                counter = 1
                while augmented_image_path.exists():
                    augmented_image_path = current_out_path / (image_name.with_stem(image_name.stem + '_' + method.__name__ + f'{counter}'))
                    counter += 1

                image = cv2.imread(category_path / image_name)
                augmented = method(image)

                cv2.imwrite(augmented_image_path, augmented)

        # copy original images
        for image_name in image_names:
            (current_out_path / image_name).write_bytes(
                (category_path / image_name).read_bytes()
            )


def augment_image(image_path: Path, dest_path: Path | None = None):
    image = cv2.imread(image_path.as_posix())
    methods = [original, rotate, blur, contrast, scale, illuminate, distort]
    
    fig, axs = plt.subplots(1, len(methods),
                            figsize=(image.shape[1] / 128 * len(methods), image.shape[0] / 128))
    for i, method in enumerate(methods):
        axs[i].imshow(method(image))
        axs[i].set_title(method.__name__)
        axs[i].axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('source', type=Path, help='path to source image or plants dataset')
    parser.add_argument('dest', type=Path, nargs='?', help='path to store augmented images')
    args = parser.parse_args()

    if args.source.is_file():
        augment_image(args.source, args.dest)
    elif args.source.is_dir():
        if args.dest is not None:
            augment_dataset(args.source, args.dest)
        else:
            raise argparse.ArgumentError(message='No destination path')
