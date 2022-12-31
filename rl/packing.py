#!/usr/bin/env python
# coding: utf-8

from vedo import settings, Line, show
import matplotlib.pyplot as plt
import numpy as np
import time

def generate_input_shells(num, size):
    input_data = np.zeros(shape=(num, size, size))
    for n in range(num):
        input_data[n] = np.random.randint(0, 2, size=(size, size))
    return input_data

def show_shells(data):
    fig, axs = plt.subplots(1, data.shape[0])
    for n in range(data.shape[0]):
        axs[n].imshow(data[n])
    plt.show()

def show_layout(data):
    fig, ax = plt.subplots()
    ax.imshow(data)
    plt.show()

#input_shells = generate_input_shells(shell_num, shell_h)
#show_shells(input_shells)

#layout = np.zeros(shape=(layout_h, layout_h))
#show_layout(layout)


def shells_bb_square(layout_data):
    # Считает площадь прямоугольника bbx of shells
    max_y = np.array(np.where(layout_data > 0))[0].max()
    min_y = np.array(np.where(layout_data > 0))[0].min()
    max_x = np.array(np.where(layout_data > 0))[1].max()
    min_x = np.array(np.where(layout_data > 0))[1].min()

    print(f"min_x:{min_x} min_y:{min_y}   max_x:{max_x} max_y:{max_y}")
    return (max_x - min_x) * (max_y - min_y)

# layout = np.zeros(shape=(layout_h, layout_h))
# for i in range(input_shells.shape[0]):
#     x = np.random.randint(0, layout_h - shell_h)
#     y = np.random.randint(0, layout_h - shell_h)
#     layout[y:y + shell_h, x:x + shell_h] += input_shells[i]
#     # break
#     # print(i)
#     show_layout(layout)
#     s = (layout > 0).sum()
#     s_bbx = shells_bb_square(layout)
#     print(f"Layout S = {layout_h ** 2}; Shells s = {s}; s bbx = {s_bbx}; Ratio(s/S) = {s / layout_h ** 2}")
#     print("Ideal Bbox (Square) = {}")



def main():
    # ----------------
    shell_h = 3  # размер квадратного ббокса
    shell_num = 5
    layout_h = 9
    # ----------------

    # !!! Включить интерактивный режим для анимации
    plt.ion()
    # Создание окна и осей для графика
    fig, ax = plt.subplots()
    layout = np.ones(shape=(layout_h, layout_h))
    print (layout.shape)
    layout = np.random.randn(layout_h, layout_h)
    print(layout.shape)
    image = ax.imshow(layout)
    shells = generate_input_shells(num=shell_num, size=shell_h)
    show_shells(shells)
    for en in range(100):

        layout = np.zeros(shape=(layout_h, layout_h))
        # random drop
        for i in range(shells.shape[0]):
            x = np.random.randint(0, layout_h - shell_h)
            y = np.random.randint(0, layout_h - shell_h)
            layout[y:y + shell_h, x:x + shell_h] += shells[i]
        #print(layout)
        #break

        image.set(data=layout)
        # Отобразить новые данный
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Задержка перед следующим обновлением
        time.sleep(1)


if __name__ == "__main__":
    main()
