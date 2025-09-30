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
    Создает изображение по умолчанию с контурами для тестирования.

    Args:
        filename (str): Имя файла для сохранения изображения по умолчанию (по умолчанию "default_image.png").

    Returns:
        np.ndarray: Созданное изображение с контурами.

    Notes:
        Создает изображение размером 300x500 с 5 кругами и 5 прямоугольниками.
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
    Загружает изображение из указанного пути или использует альтернативные источники.

    Args:
        image_path (Optional[str]): Путь к изображению (если None, используется по умолчанию или skimage.data.coins()).

    Returns:
        Optional[np.ndarray]: Загруженное изображение или None при ошибке.

    Notes:
        Если файл не найден, создает изображение по умолчанию или загружает coins() из skimage.
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
        return cv2.cvtColor(coins, cv2.COLOR_GRAY2BGR)  # Конвертация в BGR для совместимости с OpenCV

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Предобрабатывает изображение для улучшения обнаружения контуров.

    Args:
        image (np.ndarray): Исходное изображение.

    Returns:
        np.ndarray: Бинаризованное изображение после адаптивной обработки.

    Notes:
        Использует адаптивную бинаризацию для работы с разными условиями освещения.
    """
    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Адаптивная бинаризация для улучшения обнаружения в разных условиях освещения
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def validate_args(args: argparse.Namespace) -> bool:
    """
    Проверяет корректность входных аргументов.

    Args:
        args (argparse.Namespace): Объект с аргументами из командной строки.

    Returns:
        bool: True, если аргументы валидны, False в противном случае.

    Notes:
        Проверяет, что min_area не отрицательное, max_area не меньше min_area, и output в допустимом диапазоне.
    """
    if args.min_area < 0:
        print("Ошибка: Минимальная площадь (min_area) не может быть отрицательной.")
        return False
    if args.max_area < args.min_area:
        print("Ошибка: Максимальная площадь (max_area) должна быть не меньше минимальной (min_area).")
        return False
    if args.output not in ['cv2', 'plt', 'file']:
        print(f"Ошибка: Неверный режим вывода '{args.output}'. Допустимые значения: cv2, plt, file.")
        return False
    if args.image and not os.path.exists(args.image):
        print(f"Ошибка: Указанный файл изображения '{args.image}' не существует.")
        return False
    return True

def find_and_draw_contours(image: np.ndarray, min_area: float = 0, max_area: float = float('inf')) -> Tuple[np.ndarray, int]:
    """
    Находит контуры, вычисляет их площади, фильтрует по заданному диапазону и рисует отфильтрованные контуры.

    Args:
        image (np.ndarray): Исходное изображение.
        min_area (float): Минимальная площадь контура (по умолчанию 0).
        max_area (float): Максимальная площадь контура (по умолчанию бесконечность).

    Returns:
        tuple: Обработанное изображение с отфильтрованными контурами и количество оставшихся контуров.

    Notes:
        Использует аппроксимацию контуров и минимальное количество вершин для улучшения определения фигур.
    """
    image_with_contours: np.ndarray = image.copy()

    thresh = preprocess_image(image)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    
    for cnt in contours:
        # Вычисляем площадь для каждого контура
        area = cv2.contourArea(cnt)
        print(f"Контур с площадью: {area}")  # Логирование для отладки
        
        # Дополнительные проверки для улучшения определения фигур
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        # Фильтрация по площади и минимальному количеству вершин (например, >3 для фигур)
        if min_area <= area <= max_area and len(approx) >= 3:
            filtered_contours.append(cnt)

    # Рисуем отфильтрованные контуры на изображении
    cv2.drawContours(image_with_contours, filtered_contours, -1, (0, 255, 0), 2)
    return image_with_contours, len(filtered_contours)

def display_or_save_image(original_image: np.ndarray, processed_image: np.ndarray, output_mode: str) -> None:
    """
    Отображает или сохраняет исходное и обработанные изображения в зависимости от режима вывода.

    Args:
        original_image (np.ndarray): Исходное изображение.
        processed_image (np.ndarray): Обработанное изображение с контурами.
        output_mode (str): Режим вывода ('cv2', 'plt' или 'file').

    Notes:
        Комбинирует изображения для отображения или сохранения.
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

    Notes:
        Обрабатывает аргументы командной строки, проверяет их валидность и управляет выполнением программы.
    """
    parser = argparse.ArgumentParser(description="Поиск и фильтрация контуров на изображении.")
    parser.add_argument("-i", "--image", type=str, help="Путь к изображению")
    parser.add_argument("-o", "--output", type=str, choices=['cv2', 'plt', 'file'], default='cv2', help="Режим вывода: cv2, plt или file")
    parser.add_argument("-min", "--min_area", type=float, default=0, help="Минимальная площадь контура (по умолчанию 0)")
    parser.add_argument("-max", "--max_area", type=float, default=float('inf'), help="Максимальная площадь контура (по умолчанию бесконечность)")
    args = parser.parse_args()

    # Проверка входных аргументов
    if not validate_args(args):
        sys.exit(1)

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