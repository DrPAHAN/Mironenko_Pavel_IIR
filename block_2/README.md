
#  About

Код решения задачи CV_1_44 выполняет следующее:
* Загрузку или генерацию изображения и приведение в оттенки серого.
* Приминение пороговой бинеризации.
* Поиск контуров с помощью cv2.findContours().
* Выделение контуров на изображении.
* Отображение кол-ва найденных контуров."

# Installation 
```
git clone https://github.com/BilboCNet/AI_proj_CV_1_44.git
cd AI_proj_CV_1_44

pip3 install -r requirements.txt
```
# Quickstart

Для тестирования на своем изображении и выводе в формате plt (опционально cv2, file):
```
python cv_1_44_find_contours.py -i path/to/image.jpg -o plt
```
Для тестирования на сгенерированном изображении:
```
python cv_1_44_find_contours.py -o plt
```