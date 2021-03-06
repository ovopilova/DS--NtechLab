1. **task_2.ipynb** - notebook-файл с кодом тренировки. Тренировка производилась в google colab при использовании GPU. В качестве фреймворка для обучения использовался pytorch. *Большинство строчек содержит комментарии, поясняющие действия*. Для запуска необходимо запустить поочередно все ячейки и, если это необходимо, изменить путь к файлу internship_data.tar.gz в первой ячейке.

2. **process.py** - скрипт для использования нейросети, с помощью которого можно просчитать переданную через аргументы папку с изображениями. *Также представлены комментарии*


**!!!** В качестве аргумента должен передавать путь к папке с изображениями относительно папки, где находится файл process.py. Предполагается, что все изображения имеют формат jpg.

В качестве примера запустим process.py из anaconda. Папка с изображениями называется examples. Она находится в той же папке, что и process.py, поэтому в качестве аргумента при запуске надо передавать просто examples.
# Пример
![Пример](https://github.com/ovopilova/dev-intensive-2019/blob/hometask_1/anaconda.PNG)
# Результаты процессинга (файл process_results.json)
![Результат](https://github.com/ovopilova/dev-intensive-2019/blob/hometask_1/json.PNG)

3. **model_resnet50_7_steps.pth** - файл, содержащий натренированную модель. Поскольку его размер 90 мб - его не удалось добавить в гитхаб, поэтому файл необходимо скачать по следующей ссылке:


[Откуда скачать загруженную модель](https://drive.google.com/file/d/1DqaJj5tecw45eVyg_dTiTfuox1wrMT6U/view?usp=sharing)

# Используемая нейросеть

В качестве модели для тренировки была использована модель resnet50, имеющая следующую архитектуру: (fc layer был заменен в соответствии с числом классов в задаче (2))

![Архитектура resnet50](https://github.com/ovopilova/dev-intensive-2019/blob/hometask_1/resnet.png)

# Подготовка данных

Использовались следующие методы для обработки изображений:
- transforms.Compose - для объединения нескольких преобразований
- transforms.Resize - для изменения размера
- transforms.CenterCrop - для обрезки изображения по центру
- transforms.ToTensor - для преобразования в тензор
- transforms.Normalize - для стандартизации картинок с заданным средним и стандартным отклонением

# Гиперпараметры

При обучении модели были использованы следующие гиперпараметры:
- batch_size = 64 - количество примеров, используемых за одну итерацию
- lr = 0.01 - learning rate
- momentum = 0.9 - коэффициент momentum 
- weight_decay = 0.0001 - L2 регуляризация
- gamma = 0.1 - коэффициент уменьшения lr
- step_size = 5 - каждые сколько шагов lr будет умножаться на коэффициент gamma
- num_epochs = 7 - число шагов при тренировке

# Результаты

Были получены следующие результаты в результате тренировки модели:
- train loss: 0.0018 acc: 0.9997
- val loss: 0.0703 acc: 0.9827
- Best val accuracy: 0.982652 - достигнута на последнем шаге
