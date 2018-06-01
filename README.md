# Tracking System Parameters Definition

Разрабатываемая в рамках диплома система трекинга работает по принципу Valve Lighthouse ([Видео](https://youtu.be/J54dotTt7k0)) и предназначена для определения пространственного положения (ПП) сложного объекта (человека).
<p align="center"><img src="https://github.com/Jemaima/scan_system_parameters_definition/blob/master/git_imgs/cn%D1%81%D1%82%D1%80%D1%83%D0%BA%D1%82%D1%83%D1%80%D0%BD%D0%B0%D1%8F%20%D1%81%D1%85%D0%B5%D0%BC%D0%B0.jpg" align="center" width="640"><p>

Точность определения координат объекта такой системы зависит от параметров сканирующего пучка

***
## Принцип действия системы
Датчики, расположенные на объекте трекинга (ОТ), регистрируют временные задержки между световым импульсом синхронизации и временем прихода сканирующего луча. 
<p align="center">
  <img src="https://github.com/Jemaima/scan_system_parameters_definition/blob/master/git_imgs/0.png" align="center" width="360">
<p>
Так как известна угловая скорость вращения ротора, и, соответственно, сканирующего луча, а также частота счетчика, можно перевести значений этих задержек в угловое положение датчиков. 
Таким образом в качестве входных данных мы имеем угловые координаты всех видимых в данный момент времени датчиков ОТ. 
Также, заранее известно взаимное расположение всех датчиков на ОТ (U,V,W), которое определяется в момент разработки корпуса ОТ.

Файл [*object_processing.py*](https://github.com/Jemaima/scan_system_parameters_definition/blob/master/object_processing.py) содержит класс TrackdObject и реализует преобразование трехмерных координат объекта в данные, получаемые системой. (угловые координаты каждого из датчиков на ОТ). 

Считываем файл, содержащий координаты и нормали всех датчиков, задаем угол поворота и перемещения, получаем набор трехмерных координат после перемещения и поворота. Далее определяем угловые координаты.
![alt text](https://github.com/Jemaima/scan_system_parameters_definition/blob/master/git_imgs/object_processing_1.png?raw=true)

Используя *SolvePnP (OpenCV)* определяется угол поворота и перемещение трансформированного объекта.

Заданное перемещение | Определенное перемещенеи | Заданный угол поворота | Определенный  угол поворота
----------------------|----------------------|------------------|----------------
15.00, 25.00, 511.00|15.00, 25.00, 511.00|50.00, 20.00, 120.00|50.00, 20.00, 120.00

Однако на практике, значения двухмерных угловых координат проекции ОТ имеют погрешность по ряду аспектов работы системы трекинга:
* Паразитное преломление в оптических элементах  
*	Непостоянство волнового фронта 
*	Квантование по времени 
*	Биение мотора 
*	Погрешность расположения датчиков (решается калибровкой)

Для обеспечения требуемой в ТЗ погрешности системы (±1 см для определения перемещения и ±3° для определения ориентации) необходимо оценить вклад каждого этапа преобразования входных данных в суммарную погрешность определения ПП. В результате чего можно будет определить оптимальную ширину и требуемое отношение сигнал/шум сканирующего пучка, которые минимизируют погрешность определения ПП объекта и обеспечат устойчивую работу алгоритма в заданном рабочем пространстве системы.

Этапы определения погрешности системы трекинга при наличии шумовой составляющей:
1. Задать начальные трехмерные координаты датчиков объекта трекинга с мировой системе координат U,V,W. 
2. Провести серию преобразований (перемещение, вращение):

Угол поворота вокруг каждой из осей  x,y,z |	Перемещение в поперечном сечении относительно ОО базовой станции, мм	|Перемещение вдоль ОО базовой станции, мм
-----|--------|---------
от 0° до 360° с шагом 30°	|-2000 до 2000 с шагом 1000	|500, 1000, 3000, 4000, 5000, 


3. Определить проекцию объекта в плоскости, перпендикулярной ОО базовой станции используя выражение (5). 
4. Вычислить значения временных задержек
5. Добавление шумовой составляющей для значений временных координат ОТ, среднеквадратичное отклонение (СКО) которой принимает следующие значения: 10, 50, 100, 200 отсчётов или **0.2, 1.0, 2.0, 4.0 мс**
6. Перевод временных задержек в угловые координаты датчиков
7. Восстановить ПП ОТ с помощью выбранного алгоритма (X,Y,Z).
8. Оценить погрешность.

Зависимость средней погрешности определения ПП объекта по всей рабочей зоне для различных значений СКО координат проекции представлены на графике ниже

Проанализировав полученные результаты можно сделать вывод, что максимально допустимое значение погрешности определения временных задержек системой регистрации не должно превышать **2 мкc**.

Зависимости значения погрешностей определения ПП объекта в зависимости от его расположения в пространстве для различных значений СКО координат проекции ОТ представлены на рисунках ниже.



