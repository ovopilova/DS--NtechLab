from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import json
from PIL import Image
from torch.autograd import Variable
import sys

class_names = ['female', 'male'] #названия классов

path_in = os.path.abspath(str(sys.argv[1])) #считываем введеный пользователем путь к папке и ищем absolute path
device = torch.device("cpu") #выбираем cpu. Тренировалась модель на gpu в google colab
model = torch.load('model_resnet50_7_steps.pth', map_location = torch.device('cpu')) #загружаем модель
model.eval()
test_transforms = transforms.Compose([ #делаем те же преобразования с изображениями, какие были перед тренировкой модели
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def to_tensor(path_): #принимает на вход полный путь к изображению и возвращает tensor, который подается на вход функции predict_
    image = Image.open(path_) #открываем изображение 
    arr_image = np.array(image) #конвертируем в np arr
    arr_image = arr_image/255
    arr_image = np.transpose(arr_image, (2, 0, 1))
    tensor_image = torch.from_numpy(arr_image).type(torch.FloatTensor)
    
    return tensor_image

to_pil_image = transforms.ToPILImage()
def predict_(image): #принимает на вход Pillow image и возвращает номер предсказанного класса
    transform_image = test_transforms(image).float()
    transform_image = transform_image.unsqueeze_(0)
    input = Variable(transform_image)
    input = input.to(device)
    output = model(input) #предсказываем номер класса, используя загруженную ранее модель
    index = output.data.cpu().numpy().argmax()
    return index

files = os.listdir(path = path_in) #в переменную files помещаем список из названий всех имеющихся в заданной папке файлов
files_filter = filter(lambda x: x.endswith('.jpg'), files) #выбираем из них только формата jpg
images = list(files_filter) #теперь в переменной images список из названий изображений, для которых необходимо предсказать класс

d = dict() #создаем словарь, в который будем добавлять результаты (названия картинок из images будут ключами словаря)

for i in range(len(images)): #проходимся по файлам в папке и добавляем в словарь предсказанный класс
    index = predict_(to_pil_image(to_tensor(os.path.join(path_in, images[i])))) #предсказываем класс при помощи
                                                                                                            #ранее введеной функции
    d[images[i]] = str(class_names[index]) #добавляем в словарь название класса в зависимости от предсказанной метки (0 или 1)

with open('process_results.json', 'w') as fp: #сохраняем результаты в файл process_results.json
    json.dump(d, fp)
print("Результаты работы нейросети сохранены в файл process_results.json")