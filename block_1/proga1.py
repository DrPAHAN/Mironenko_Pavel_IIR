import argparse
import cv2
import sys

class ShapeDetector:
    def __init__(self):
        """Инициализирует детектор фигур."""
        pass

    def detect(self, c):
        """Определяет форму контура на основе количества вершин и соотношения сторон.

        Args:
            c (numpy.ndarray): Контур фигуры.

        Returns:
            str: Название фигуры ('прямоугольник', 'квадрат' или 'неопознанная').
        """
        # Инициализируем имя фигуры и аппроксимируем контур
        shape = "неопознанная"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # Если фигура имеет 4 вершины, это либо квадрат, либо прямоугольник
        if len(approx) == 4:
            # Вычисляем ограничивающий прямоугольник контура и используем его для вычисления соотношения сторон
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # Квадрат будет иметь соотношение сторон, близкое к единице, иначе это прямоугольник
            shape = "квадрат" if ar >= 0.95 and ar <= 1.05 else "прямоугольник"

        return shape

def resize_image(image, target_width=300):
    """Масштабирует изображение до заданной ширины с сохранением пропорций.

    Args:
        image (numpy.ndarray): Исходное изображение.
        target_width (int): Целевая ширина (по умолчанию 300).

    Returns:
        tuple: Измененное изображение и коэффициент масштабирования.
    """
    ratio = image.shape[0] / float(image.shape[1])
    resized = cv2.resize(image, (target_width, int(target_width * ratio)))
    return resized, image.shape[0] / float(resized.shape[0])

def binarize_image(image, threshold):
    """Преобразует изображение в бинарную форму: градации серого, размытие и порог.

    Args:
        image (numpy.ndarray): Исходное изображение.
        threshold (int): Порог для бинаризации.

    Returns:
        numpy.ndarray: Бинаризованное изображение.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]

def find_contours(image):
    """Находит контуры в бинаризованном изображении.

    Args:
        image (numpy.ndarray): Бинаризованное изображение.

    Returns:
        list: Список контуров.
    """
    return cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

def main():
    """Основная функция для обнаружения прямоугольных объектов на изображении.

    Обрабатывает изображение, определяет прямоугольные объекты и визуализирует результат.
    """
    # Создаем парсер аргументов и парсим аргументы
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="путь к входному изображению")
    ap.add_argument("-t", "--threshold", type=int, default=60,
                    help="порог для бинаризации (по умолчанию 60)")
    ap.add_argument("-o", "--output", type=str,
                    help="путь для сохранения результирующего изображения")
    args = vars(ap.parse_args())

    # Проверяем корректность порога
    if args["threshold"] < 0:
        print("Ошибка: Порог должен быть неотрицательным числом.")
        sys.exit(1)

    # Загружаем изображение и обрабатываем возможные ошибки
    try:
        image = cv2.imread(args["image"])
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {args['image']}")
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        sys.exit(1)

    # Масштабируем изображение
    resized, ratio = resize_image(image)

    # Бинаризуем изображение
    thresh = binarize_image(resized, args["threshold"])

    # Находим контуры
    contours = find_contours(thresh)

    sd = ShapeDetector()

    rectangles_found = 0
    output_image = image.copy()

    # Проходим по всем контурам
    for c in contours:
        # Вычисляем центр контура, затем определяем имя фигуры только по контуру
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)

        # Обрабатываем только прямоугольники и квадраты
        if shape in ["прямоугольник", "квадрат"]:
            rectangles_found += 1
            # Умножаем координаты контура (x, y) на коэффициент масштабирования,
            # затем рисуем контуры и имя фигуры на изображении
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
            cv2.putText(output_image, "rectangle", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

    # Показываем и сохраняем результирующее изображение
    cv2.imshow("Изображение", output_image)
    if args["output"]:
        cv2.imwrite(args["output"], output_image)
        print(f"Результирующее изображение сохранено как: {args['output']}")
    cv2.waitKey(0)

    print(f"Найдено {rectangles_found} прямоугольных объектов.")

if __name__ == "__main__":
    main()