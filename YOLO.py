Main.py
from imageai.Detection import VideoObjectDetection
import os
import tensorflow
import imageai
import keras
import matplotlib
from matplotlib import pyplot as plt
from tkinter import messagebox


messagebox.showwarning("Предупреждение","Данное программное обеспечение ресурсозатратное. Возможно будут подлагивания приложения и повыение температуры ЦП.")
messagebox.showinfo("Предварительная информация","Проверьте. что вы загрузили в корневую папку с программой видео для анализа с наименованием'For_Analysis'формата .mp4.")

execution_path = os.getcwd()
color_index = {'bus': 'red',   'truck': 'indigo', 'motorcycle': 'azure', 'bicycle': 'olivedrab',  'train': 'cornsilk', 'car': 'silver' , 'person': 'honeydew'}

resized = False

def forFrame(frame_number, output_array, output_count, returned_frame):

    plt.clf()

    this_colors = []
    labels = []
    sizes = []

    counter = 0

    for eachItem in output_count:
        counter += 1
        labels.append(eachItem + " = " + str(output_count[eachItem]))
        sizes.append(output_count[eachItem])
        this_colors.append(color_index[eachItem])

    global resized

    if (resized == False):
        manager = plt.get_current_fig_manager()
        manager.resize(width=1920, height=1080)
        resized = True

    plt.subplot(1, 2, 1)
    plt.title("Кадр : " + str(frame_number))
    plt.axis("off")
    plt.imshow(returned_frame, interpolation="none")

    plt.subplot(1, 2, 2)
    plt.title("Статистика по кадру: " + str(frame_number))
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct="%1.1f%%")

    plt.pause(0.01)



detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel()

plt.show()

custom_objects = detector.CustomObjects(car=True, motorcycle=True, bus = True, train=True, person=True, truck=True,    bicycle=True, clock=False)
video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects,
   input_file_path=os.path.join(execution_path, "For_Analysis.mp4"),
   output_file_path=os.path.join(execution_path, "video_second_analysis"),
   frames_per_second=20,
    minimum_percentage_probability=30,
   display_percentage_probability=False,
    per_frame_function=forFrame,
    return_detected_frame=True,
   log_progress=True
)
messagebox.showinfo("Завершение работы программы","Программа обработала файл. Для обработки следующего файла запустите программу повторно, после охлаждения ЦП.")
os.system("pause")
