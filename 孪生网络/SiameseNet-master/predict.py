# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2020/5/20 10:23
# @Software: PyCharm
# @Brief: 模型预测

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from main import siamese_network
import config as cfg
import cv2 as cv
from random import sample


def main():
    # # left_path = "./dataset/images_background/Arcadian/character07/0007_11.png"
    # left_path = "./dataset/images_background/Alphabet_of_the_Magi/character01/0709_02.png"
    # # right_path = "./dataset/images_background/Arcadian/character07/0007_03.png"
    # right_path = "./dataset/images_background/Alphabet_of_the_Magi/character01/0709_03.png"
    # left_path = "./dataset/images_evaluation/Angelic/character09/0973_01.png"
    # right_path = "./dataset/images_evaluation/Angelic/character10/0974_05.png"

    base_dir = "./dataset/images_evaluation/"


    g = os.walk(base_dir)
    group = []
    pic_result = []

    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            if len(os.path.join(path, dir_name).replace("\\","/").split("/")) == 4:
                group.append(os.path.join(path, dir_name))
    group_list = sample(group,10)

    character_list = []
    for i in range(1,21):
        character_list.append("character" + str(i).zfill(2))
    character_list_1 = sample(character_list,10)
    pic_dir = []
    for group_ in group_list:
        for character_ in character_list_1:
            pic_dir.append(os.path.join(group_,character_).replace("\\","/"))
    s = ["_"+str(i).zfill(2) for i in range(1,21)]
    num_list = sample(s,10)
    for pic_ in pic_dir:
        pic = []
        for path, dir_list, file_list in os.walk(pic_):
            for file in file_list:
                for num_ in num_list:
                    if file.find(num_) != -1:
                        pic.append(os.path.join(pic_,file).replace("\\","/"))
                        break
        pic_result.append(pic)
    evaluation_list = {"true":[],"false":[]}
    for pic_re in pic_result:
        for i in range(0,len(pic_re),2):
            left_img = pic_re[i]
            right_img = pic_re[i+1]
            evaluation_list["true"].append({"left_img":left_img,"right_img":right_img})

    for i in range(0,len(pic_result),2):
        for s in zip(pic_result[i],pic_result[i+1]):
            left_img = s[0]
            right_img = s[1]
            evaluation_list["false"].append({"left_img":left_img,"right_img":right_img})


    count = 0
    model = siamese_network()
    model.load_weights(cfg.model_path)
    for h_ in evaluation_list:
        for img in evaluation_list[h_]:
            left_path = img["left_img"]
            right_path = img["right_img"]

            left_img = cv.imread(left_path, cv.IMREAD_GRAYSCALE)
            right_img = cv.imread(right_path, cv.IMREAD_GRAYSCALE)


            left_img = tf.expand_dims(left_img, axis=0)
            left_img = tf.expand_dims(left_img, axis=-1)
            right_img = tf.expand_dims(right_img, axis=0)
            right_img = tf.expand_dims(right_img, axis=-1)

            left_img = tf.cast(left_img, tf.float32)
            right_img = tf.cast(right_img, tf.float32)

            result = model.predict([left_img, right_img],steps=1)

            if result > cfg.similar_threshold:
                kkk = "true"
            else:
                kkk = "false"

            print(kkk)
            if kkk == h_:
                count += 1
    print(count/1000)


if __name__ == '__main__':
    main()
