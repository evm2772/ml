# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import numpy


def gaussian(x, delay, sigma):
    '''
    Функция, график которой будет отображаться процессе анимации
    '''
    return numpy.exp(-((x - delay) / sigma) ** 2)


if __name__ == '__main__':
    # Параметры отображаемой функции
    maxSize = 200
    sigma = 10.0

    # Диапазон точек для расчета графика функции
    x = numpy.arange(maxSize)

    # Значения графика функции
    y = numpy.zeros(maxSize)

    # !!! Включить интерактивный режим для анимации
    plt.ion()

    # Создание окна и осей для графика
    fig, ax = plt.subplots()

    # Установка отображаемых интервалов по осям
    ax.set_xlim(0, maxSize)
    ax.set_ylim(-1.1, 1.1)

    # Отобразить график фукнции в начальный момент времени
    line, = ax.plot(x, y)

    # У функции gaussian будет меняться параметр delay (задержка)
    for delay in numpy.arange(-50.0, 200.0, 1.0):
        y = gaussian(x, delay, sigma)

        # Обновить данные на графике
        line.set_ydata(y)

        # Отобразить новые данный
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Задержка перед следующим обновлением
        time.sleep(0.02)

    # Отключить интерактивный режим по завершению анимации
    plt.ioff()
    plt.show()