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
<p align="center">
  <img src="https://github.com/Jemaima/scan_system_parameters_definition/blob/master/git_imgs/noise_dep.png" width="480">
<p>

Проанализировав полученные результаты можно сделать вывод, что максимально допустимое значение погрешности определения временных задержек системой регистрации не должно превышать **2 мкc**.

Зависимости погрешности определения перемещения и угловой ориентации объекта в зависимости от его положения в рабочем пространстве (от 0.5 до 5 м в продольном направлении и от -2 до 2 м в поперечном) для различных значений СКО шума представлены на рисунках ниже.
<p align="center">
  <img src="https://github.com/Jemaima/scan_system_parameters_definition/blob/master/git_imgs/t_error.png" "Перемещение" width="360">
&#160; &#160; &#160; &#160; &#160; &#160;
  <img src="https://github.com/Jemaima/scan_system_parameters_definition/blob/master/git_imgs/r_error.png" width="360">   
<p>

На следующем этапе необходимо определить такие параметры сигнала (импульса) как длительность и отношение сигнал/шум. Длительность импульса определяется шириной пучка и частотой вращений ротора.  
Очевидно, что длительность импульса при постоянной ширине сканирующего пучка будет зависеть от положения датчика в рабочем пространстве. Чем датчик дальше от сисемы подсвета, тем быстрее сканирующий пучок его проходит, соответственно, длительность импульса меньше (та же зависимость при изменении угла поворота).

Параметры моделирования:

Дальность до объекта,  м	| Угловая ориентация датчика относительно ОО системы подсвета, °	| Относительная величина шумовой составляющей |	Ширина сканирующего пучка, мм
------|------|------|------
от 0.5 до 5 с шагом 0.5 | От 0 до 60 с шагом 15° | От 0.1 до 0.7 с шагом 0.1 | От 10 до 70 с шагом 10

Результат моделирования представлен на рисунке ниже, где отмечена зона параметров, удовлетворяющих заданной погрешности. 

**Таким образом, оптимальной шириной сканирующего пучка будем считать 20 мм. При этом необходимо, чтобы освещенность в плоскости ПИ обеспечивала отношение сигнал/шум не менее 5 в любой точке рабочего пространства.**
<p align="center">
  <img src="https://github.com/Jemaima/scan_system_parameters_definition/blob/master/git_imgs/6KeIKKouGgk.jpg" width="360">   
<p>





