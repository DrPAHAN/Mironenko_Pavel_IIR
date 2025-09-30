import os
import gc
import sys
import argparse
import cv2
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from skimage import data

def create_default_image(filename: str = "default_image.png") -> np.ndarray:
    """
    Создает изображение по умолчанию с контурами.
    """
    print(f"Файл '{filename}' не найден. Создание изображения по умолчанию.")
    image: np.ndarray = np.zeros((300, 500, 3), dtype="uint8")
    
    # Рисуем 10 фигур (5 кругов, 5 прямоугольников)
    for i in range(5):
        cv2.circle(image, (60 + i * 90, 75), 30, (255, 255, 255), -1)
        cv2.rectangle(image, (30 + i * 90, 175), (90 + i * 90, 235), (255, 255, 255), -1)

    cv2.imwrite(filename, image)
    return image

def load_image(image_path: Optional[str]) -> Optional[np.ndarray]:
    """
    Загружает изображение из указанного пути.
    Если путь не указан или файл отсутствует, создает изображение по умолчанию или использует skimage.data.coins().
    """
    default_filename: str = "default_image.png"

    if image_path:
        if not os.path.exists(image_path):
            print(f"Ошибка: файл '{image_path}' не найден.")
            return None
        return cv2.imread(image_path)

    if os.path.exists(default_filename):
        print(f"Использование изображения по умолчанию: '{default_filename}'")
        return cv2.imread(default_filename)
    else:
        print("Загрузка изображения монет из skimage.data.coins()")
        coins = data.coins()
        return cv2.cvtColor(coins, cv2.COLOR_RGB2BGR)  # Конвертация для совместимости с OpenCV

def find_and_draw_contours(image: np.ndarray, min_area: float = 0, max_area: float = float('inf')) -> Tuple[np.ndarray, int]:
    """
    Находит контуры, вычисляет их площади, фильтрует по заданному диапазону и рисует отфильтрованные контуры.

    Args:
        image (np.ndarray): Исходное изображение.
        min_area (float): Минимальная площадь контура (по умолчанию 0).
        max_area (float): Максимальная площадь контура (по умолчанию бесконечность).

    Returns:
        tuple: Обработанное изображение с отфильтрованными контурами и количество оставшихся контуров.
    """
    image_with_contours: np.ndarray = image.copy()

    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    
    for cnt in contours:
        # Вычисляем площадь для каждого контура
        area = cv2.contourArea(cnt)
        print(f"Контур с площадью: {area}")  # Логирование для отладки
        # Оставляем только контуры в заданном диапазоне площадей
        if min_area <= area <= max_area:
            filtered_contours.append(cnt)

    # Рисуем отфильтрованные контуры на изображении
    cv2.drawContours(image_with_contours, filtered_contours, -1, (0, 255, 0), 2)
    return image_with_contours, len(filtered_contours)

def display_or_save_image(original_image: np.ndarray, processed_image: np.ndarray, output_mode: str) -> None:
    """
    Отображает или сохраняет исходное и обработанные изображения в зависимости от режима вывода.
    """
    if output_mode == 'cv2':
        combined_image = np.hstack((original_image, processed_image))
        cv2.imshow("Original | Contours", combined_image)
        cv2.waitKey(0)
    elif output_mode == 'plt':
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(processed_rgb)
        axes[1].set_title("Filtered Contours")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
    elif output_mode == 'file':
        combined_image = np.hstack((original_image, processed_image))
        output_filename = "result.png"
        cv2.imwrite(output_filename, combined_image)
        print(f"Комбинированное изображение сохранено как: '{output_filename}'")

def main() -> None:
    """
    Инициализирует приложение для обнаружения и фильтрации контуров.
    """
    parser = argparse.ArgumentParser(description="Поиск и фильтрация контуров на изображении.")
    parser.add_argument("-i", "--image", type=str, help="Путь к изображению")
    parser.add_argument("-o", "--output", type=str, choices=['cv2', 'plt', 'file'], default='cv2', help="Режим вывода: cv2, plt или file")
    parser.add_argument("-min", "--min_area", type=float, default=0, help="Минимальная площадь контура (по умолчанию 0)")
    parser.add_argument("-max", "--max_area", type=float, default=float('inf'), help="Максимальная площадь контура (по умолчанию бесконечность)")
    args = parser.parse_args()

    original_image: Optional[np.ndarray] = load_image(args.image)

    if original_image is None:
        return

    processed_image = None
    try:
        processed_image, num_contours = find_and_draw_contours(original_image, args.min_area, args.max_area)
        # Вывод количества оставшихся контуров
        print(f"Количество оставшихся контуров: {num_contours}")

        text = f"Contours found: {num_contours}"
        cv2.putText(processed_image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        display_or_save_image(original_image, processed_image, args.output)

    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        try:
            del original_image
            if processed_image is not None:
                del processed_image
        except NameError:
            pass
        gc.collect()

if __name__ == "__main__":
    main()