import argparse
import cv2

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
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
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        return shape

# Создаем парсер аргументов и парсим аргументы
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="путь к входному изображению")
args = vars(ap.parse_args())

# Загружаем изображение и изменяем его размер для лучшей аппроксимации фигур
image = cv2.imread(args["image"])
resized = cv2.resize(image, (300, int(300 * image.shape[0] / image.shape[1])))
ratio = image.shape[0] / float(resized.shape[0])

# Преобразуем уменьшенное изображение в градации серого, слегка размыкаем его и применяем порог
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# Находим контуры в пороговом изображении
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

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
    if shape in ["rectangle", "square"]:
        rectangles_found += 1
        # Умножаем координаты контура (x, y) на коэффициент масштабирования,
        # затем рисуем контуры и имя фигуры на изображении
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2)
        cv2.putText(output_image, "rectangle", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2)

# Показываем результирующее изображение
cv2.imshow("Изображение", output_image)
cv2.waitKey(0)

print(f"Найдено {rectangles_found} прямоугольных объектов.")