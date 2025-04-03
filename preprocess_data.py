from shutil import rmtree
from pathlib import Path

from augmentation import augment_dataset
from transformation import apply_transformations


if __name__ == '__main__':
    if Path('augmented_images').exists():
        print('Delete directory \'augmented_images\'')
    if Path('processed_images').exists():
        print('Delete directory \'processed_images\'')
    
    # augment
    augment_dataset(Path('images/Apple'), Path('augmented_images/Apple'))
    augment_dataset(Path('images/Grape'), Path('augmented_images/Grape'))

    # transform
    apply_transformations('augmented_images/Apple/Apple_Black_rot', 'processed_images/Apple/Apple_Black_rot')
    apply_transformations('augmented_images/Apple/Apple_healthy',   'processed_images/Apple/Apple_healthy')
    apply_transformations('augmented_images/Apple/Apple_rust',      'processed_images/Apple/Apple_rust')
    apply_transformations('augmented_images/Apple/Apple_scab',      'processed_images/Apple/Apple_scab')

    apply_transformations('augmented_images/Grape/Grape_Black_rot', 'processed_images/Grape/Grape_Black_rot')
    apply_transformations('augmented_images/Grape/Grape_Esca',      'processed_images/Grape/Grape_Esca')
    apply_transformations('augmented_images/Grape/Grape_healthy',   'processed_images/Grape/Grape_healthy')
    apply_transformations('augmented_images/Grape/Grape_spot',      'processed_images/Grape/Grape_spot')

    rmtree('augmented_images')
