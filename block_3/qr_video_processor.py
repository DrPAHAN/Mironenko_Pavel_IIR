import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import pyzbar.pyzbar as pyzbar
import qrcode

def file_add(name_f: str, qr_cd: str) -> None:
    """Создает QR-код и сохраняет его в указанный файл.

    Args:
        name_f (str): Путь к файлу, в который будет сохранен QR-код.
        qr_cd (str): Данные для создания QR-кода.
    """
    img = qrcode.make(qr_cd)
    img.save(name_f)

def Dec(name_f: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Декодирует QR-код из изображения.

    Args:
        name_f (str): Путь к файлу изображения с QR-кодом.

    Returns:
        tuple[str, np.ndarray, np.ndarray]: Кортеж из данных QR-кода, координат bounding box и изображения.
    """
    img = cv2.imread(name_f)
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)
    return data, bbox, img

def write_p(data: str, bbox: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Рисует boundingrelated: bounding box вокруг QR-кода и данные на изображении, сохраняет результат.

    Args:
        data (str): Данные, извлеченные из QR-кода.
        bbox (np.ndarray): Координаты bounding box QR-кода.
        img (np.ndarray): Входное изображение.

    Returns:
        np.ndarray: Изображение с нарисованным bounding box и данными QR-кода.
    """
    if bbox is not None:
        print(f"Данные QR-кода: {data}")
        points = [(int(point[0]), int(point[1])) for point in bbox[0]]
        print(f"Координаты: {points}")
        for i in range(4):
            cv2.line(img, points[i], points[(i + 1) % 4], (255, 255, 0), 10)
    cv2.imwrite("img.png", img)
    return img

def parse_arguments() -> argparse.Namespace:
    """Парсит аргументы командной строки для обработки видео.

    Returns:
        argparse.Namespace: Объект с аргументами командной строки.
    """
    parser = argparse.ArgumentParser(description='Обработка видео с поиском QR-кода')
    parser.add_argument("--video_path", "-v", type=str, required=True, help="Путь к входному видеофайлу")
    parser.add_argument("--output_folder", "-o", type=str, default="out", help="Папка для сохранения результатов")
    parser.add_argument("--out_name", "-n", type=str, default="output", help="Имя выходного файла")
    parser.add_argument("--log_file", "-l", type=str, default="qr_log.txt", help="Имя файла лога")
    return parser.parse_args()

def validate_arguments(args: argparse.Namespace) -> bool:
    """Проверяет существование входного видеофайла.

    Args:
        args (argparse.Namespace): Аргументы командной строки.

    Returns:
        bool: True, если файл существует, иначе False.
    """
    if not os.path.exists(args.video_path):
        print(f"Ошибка: Видеофайл '{args.video_path}' не найден!")
        return False
    return True

def rotate_template(template: np.ndarray, angle: int) -> np.ndarray:
    """Поворачивает шаблонное изображение на указанный угол.

    Args:
        template (np.ndarray): Исходное изображение шаблона.
        angle (int): Угол поворота (0, 90, 180, 270 градусов).

    Returns:
        np.ndarray: Повернутое изображение шаблона.
    """
    if angle == 0:
        return template.copy()
    elif angle == 90:
        return cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(template, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return template.copy()

def non_max_suppression(boxes: list, scores: list, iou_threshold: float = 0.3, score_threshold: float = 0.5) -> list:
    """Применяет алгоритм подавления немаксимумов для выбора лучших bounding box.

    Args:
        boxes (list): Список координат bounding box (x, y, w, h).
        scores (list): Список значений уверенности для каждого bounding box.
        iou_threshold (float, optional): Порог IoU для подавления. По умолчанию 0.3.
        score_threshold (float, optional): Порог уверенности. По умолчанию 0.5.

    Returns:
        list: Индексы выбранных bounding box.
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    above_threshold = scores >= score_threshold
    boxes = boxes[above_threshold]
    scores = scores[above_threshold]
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        union = areas[i] + areas[indices[1:]] - intersection
        iou = intersection / union
        remaining_indices = np.where(iou <= iou_threshold)[0]
        indices = indices[remaining_indices + 1]
    return keep

def searching_single_obj_with_rotation(frame: np.ndarray, template_gray_orig: np.ndarray, threshold: float = 0.65,
                                       scale_range: tuple = (0.3, 1.0), steps: int = 5) -> tuple[np.ndarray, tuple, float]:
    """Ищет объект в кадре с учетом поворотов и масштабирования.

    Args:
        frame (np.ndarray): Входной кадр видео.
        template_gray_orig (np.ndarray): Шаблонное изображение в градациях серого.
        threshold (float, optional): Порог уверенности для детекции. По умолчанию 0.65.
        scale_range (tuple, optional): Диапазон масштабирования. По умолчанию (0.3, 1.0).
        steps (int, optional): Количество шагов масштабирования. По умолчанию 5.

    Returns:
        tuple[np.ndarray, tuple, float]: Обработанный кадр, координаты bounding box (x, y, w, h), уверенность.
    """
    try:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = img_gray.shape[:2]
        result_frame = frame.copy()
        method = cv2.TM_CCOEFF_NORMED
        best_val = -1
        best_loc = None
        best_size = None
        best_angle = 0
        found = False
        rotation_angles = [0, 90, 180, 270]
        for angle in rotation_angles:
            rotated_template = rotate_template(template_gray_orig, angle)
            templ_h, templ_w = rotated_template.shape[:2]
            scales = np.linspace(scale_range[0], scale_range[1], steps)[::-1]
            for scale in scales:
                new_w = int(templ_w * scale)
                new_h = int(templ_h * scale)
                if new_w > frame_w or new_h > frame_h or new_w < 10 or new_h < 10:
                    continue
                resized_template = cv2.resize(rotated_template, (new_w, new_h), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(img_gray, resized_template, method)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_size = (new_w, new_h)
                    best_angle = angle
                    found = True
        if found:
            top_left = best_loc
            bottom_right = (top_left[0] + best_size[0], top_left[1] + best_size[1])
            cv2.rectangle(result_frame, top_left, bottom_right, (0, 255, 0), 3)
            info_text = f'Conf: {best_val:.2f}'
            cv2.putText(result_frame, info_text, (top_left[0], max(0, top_left[1] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return result_frame, (top_left[0], top_left[1], best_size[0], best_size[1]), best_val
        return result_frame, None, 0
    except Exception as e:
        print(f"Ошибка при отслеживании: {e}")
        return frame, None, 0

def images_to_video(image_folder: str, video_path: str, fps: float, codec: str = 'mp4v') -> bool:
    """Собирает изображения из папки в видео и сохраняет его.

    Args:
        image_folder (str): Путь к папке с изображениями.
        video_path (str): Путь для сохранения выходного видео.
        fps (float): Частота кадров выходного видео.
        codec (str, optional): Кодек для видео. По умолчанию 'mp4v'.

    Returns:
        bool: True, если видео успешно сохранено, иначе False.
    """
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))])
    if not images:
        print(f"Ошибка: Нет изображений в папке '{image_folder}' для сборки видео!")
        return False
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Ошибка: Не удалось загрузить изображение '{first_image_path}'!")
        return False
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not video.isOpened():
        print(f"Ошибка: Не удалось создать видеофайл '{video_path}'! Проверьте права доступа или кодек.")
        return False
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"Предупреждение: Не удалось загрузить изображение '{img_path}'")
    video.release()
    print(f"Видео успешно сохранено: {video_path}")
    return True

def clean_up(image_folder: str) -> None:
    """Удаляет временные изображения из указанной папки.

    Args:
        image_folder (str): Путь к папке с изображениями.
    """
    for img in os.listdir(image_folder):
        if img.endswith(".jpg") or img.endswith(".png"):
            img_path = os.path.join(image_folder, img)
            try:
                os.remove(img_path)
            except Exception as e:
                print(f"Не удалось удалить {img_path}: {e}")

def process_video(video_path: str, output_folder: str, out_name: str, log_file: str) -> int:
    """Обрабатывает видео, ищет QR-коды, сохраняет обработанные кадры и видео.

    Args:
        video_path (str): Путь к входному видеофайлу.
        output_folder (str): Папка для сохранения результатов.
        out_name (str): Имя выходного видеофайла (без расширения).
        log_file (str): Путь к файлу лога.

    Returns:
        int: Количество кадров с обнаруженными QR-кодами.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл '{video_path}'!")
        return 0
    frame_count = 0
    qr_detected_frames = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log = []
    
    temp_qr_file = "temp_qr.png"
    file_add(temp_qr_file, "https://example.com")
    template = cv2.imread(temp_qr_file, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Ошибка: Не удалось загрузить шаблон QR-кода '{temp_qr_file}'!")
        cap.release()
        return 0
    
    detector = cv2.QRCodeDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        data, bbox, _ = detector.detectAndDecode(frame)
        result_frame = frame.copy()
        log_entry = {"frame": frame_count, "data": None, "bbox": None, "confidence": 0}
        
        if data:
            qr_detected_frames += 1
            log_entry["data"] = data
            if bbox is not None:
                points = [(int(point[0]), int(point[1])) for point in bbox[0]]
                log_entry["bbox"] = points
                for i in range(4):
                    cv2.line(result_frame, points[i], points[(i + 1) % 4], (255, 255, 0), 3)
                cv2.putText(result_frame, data, (points[0][0], points[0][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            result_frame, bbox, confidence = searching_single_obj_with_rotation(frame, template)
            if bbox:
                log_entry["bbox"] = bbox
                log_entry["confidence"] = confidence
                x, y, w, h = bbox
                roi = frame[y:y+h, x:x+w]
                data, _, _ = detector.detectAndDecode(roi)
                if data:
                    qr_detected_frames += 1
                    log_entry["data"] = data
                    cv2.putText(result_frame, data, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        print(f"Кадр {frame_count}: Данные: {log_entry['data']}, Координаты: {log_entry['bbox']}, Уверенность: {log_entry['confidence']:.2f}")
        log.append(log_entry)
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
        if not cv2.imwrite(frame_filename, result_frame):
            print(f"Ошибка: Не удалось сохранить кадр '{frame_filename}'!")
        
        frame_count += 1
    
    with open(log_file, 'a') as f:
        f.write(f"Обработка видео: {video_path}, {datetime.now()}\n")
        f.write(f"Всего кадров: {frame_count}, Кадров с QR-кодом: {qr_detected_frames}\n")
        for entry in log:
            f.write(f"Кадр {entry['frame']}: Данные: {entry['data']}, Координаты: {entry['bbox']}, Уверенность: {entry['confidence']:.2f}\n")
    
    print(f"Кадров с распознанным QR-кодом: {qr_detected_frames}")
    if qr_detected_frames > 3:
        print("Метрика выполнена: QR-код распознан в более чем 3 кадрах")
    else:
        print("Метрика не выполнена: QR-код распознан в менее чем 4 кадрах")
    
    video_output_path = os.path.join(output_folder, f"{out_name}.mp4")
    success = images_to_video(output_folder, video_output_path, fps)
    if success:
        clean_up(output_folder)
    else:
        print(f"Ошибка: Не удалось создать видео '{video_output_path}'")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if os.path.exists(temp_qr_file):
        try:
            os.remove(temp_qr_file)
        except Exception as e:
            print(f"Не удалось удалить временный файл '{temp_qr_file}': {e}")
    
    return qr_detected_frames

def main() -> None:
    """Основная функция для запуска обработки видео."""
    args = parse_arguments()
    if not validate_arguments(args):
        print("Ошибка в аргументах. Пример: python script.py -v video.mp4 -o out -n output -l qr_log.txt")
        exit(1)
    
    print("Начинаем обработку видео...")
    qr_detected_frames = process_video(args.video_path, args.output_folder, args.out_name, args.log_file)
    
    if qr_detected_frames:
        print(f"Обработка завершена. Результаты сохранены в {args.output_folder}")
    else:
        print("Ошибка при обработке видео")

if __name__ == "__main__":
    main()