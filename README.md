# Tracking System Parameters Definition

Разрабатываемая в рамках диплома система трекинга работает по принципу Valve Lighthouse ([Видео](https://youtu.be/J54dotTt7k0)).
![alt text](https://github.com/Jemaima/scan_system_parameters_definition/blob/master/git_imgs/cn%D1%81%D1%82%D1%80%D1%83%D0%BA%D1%82%D1%83%D1%80%D0%BD%D0%B0%D1%8F%20%D1%81%D1%85%D0%B5%D0%BC%D0%B0.jpg?raw=true)
Точность определения координат объекта такой системы зависит от параметров сканирующего пучка

***
## Принцип действия системы
Датчики, расположенные на объекте трекинга (ОТ), регистрируют временные задержки между световым импульсом синхронизации и временем прихода сканирующего луча. 


Так как известна угловая скорость вращения ротора, и, соответственно, сканирующего луча, а также частота счетчика, можно перевести значений этих задержек в угловое положение датчиков. 
Таким образом в качестве входных данных мы имеем угловые координаты всех видимых в данный момент времени датчиков ОТ. 
Также, заранее известно взаимное расположение всех датчиков на ОТ (U,V,W), которое определяется в момент разработки корпуса ОТ.

Файл [object_processing.py](https://github.com/Jemaima/scan_system_parameters_definition/blob/master/object_processing.py) содержит класс TrackdObject и реализует преобразование трехмерных координат объекта в данные, получаемые системой. (угловые координаты каждого из датчиков на ОТ). 

Считываем файл, содержащий координаты и нормали всех датчиков, задаем угол поворота и перемещения, получаем набор трехмерных координат после перемещения и поворота. Далее определяем угловые координаты.
![alt text](https://github.com/Jemaima/scan_system_parameters_definition/blob/master/git_imgs/object_processing_1.png?raw=true)

Используя SolvePnP (OpenCV) определяется угол поворота и перемещение трансформированного объекта.

Заданное перемещение | Определенное перемещенеи | Заданный угол поворота | Определенный  угол поворота
----------------------|----------------------|------------------|----------------
15.00, 25.00, 511.00|15.00, 25.00, 511.00|50.00, 20.00, 120.00|50.00, 20.00, 120.00



