import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def analyze_images(directory, plant_name):
    plant_diseases = []
    for subdir, dirs, files in os.walk(directory):
        if subdir == directory:
            continue
        disease_name = os.path.basename(subdir)
        image_files = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        if plant_name.lower() in disease_name.lower() and image_files:
            plant_diseases.append((disease_name, len(image_files)))
    return plant_diseases

def plot_pie_chart(data, title):
    if not data:
        print(f"Нет данных для {title}!")
        return

    labels, sizes = zip(*data)

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f'Распределение заболеваний растений - {title}')
    plt.axis('equal')
    plt.show()

def plot_bar_chart(data, title):
    if not data:
        print(f"Нет данных для {title}!")
        return

    labels, sizes = zip(*data)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes)
    plt.xlabel('Тип заболевания')
    plt.ylabel('Количество изображений')
    plt.title(f'Распределение заболеваний растений - {title}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main(directory):
    plant_names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

    for plant_name in plant_names:
        print(f"Обрабатываем данные для {plant_name}...")
        plant_diseases = analyze_images(directory, plant_name)

        if not plant_diseases:
            print(f"Не найдено заболеваний для растения: {plant_name}")
            continue

        plot_pie_chart(plant_diseases, plant_name)
        plot_bar_chart(plant_diseases, plant_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distribution')
    parser.add_argument('source', type=Path, help='path to plants dataset')
    args = parser.parse_args()

    main(args.source)