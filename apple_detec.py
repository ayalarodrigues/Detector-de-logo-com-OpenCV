import dlib
import glob
import os
import cv2

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True #realiza mudanças nas imagens (ex: inclinação)
options.C = 5 #custo do algoritmo SVM (quanto maior C, maior a margem de erro = melhores resultados 5 é padrão da doc)

dlib.train_simple_object_detector("recursos/treinamento_apple.xml", "recursos/detector_apple.svm", options)
#realiza o treinamento utilizando as opções acima

