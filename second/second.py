import numpy as np


def act(x):
    """
    function activation
    :param x:
    :return:
    """
    return 0 if x < 0.5 else 1


def go(house, rock, attr):
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12])  # матрица 2x3
    weight2 = np.array([-1, 1])  # вектор 1х3

    sum_hidden = np.dot(weight1, x)  # вычисляем сумму на входах нейронов скрытого слоя
    print(f"Значения сумм на нейронах скрытого слоя: {sum_hidden}")

    out_hidden = np.array([act(x) for x in sum_hidden])
    print(f"Значения на выходах нейронов скрытого слоя: {out_hidden}")

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print(f"Выходное значение НС: {y}")

    return y


def log_result(house, rock, attr):
    res_string = 'Вводные данные: \n' \
                 f'Дом = {house}\n' \
                 f'Отношение к тяжелому року = {rock}\n' \
                 f'Красота = {attr}'
    print('*********************************************************')
    print(res_string)
    print('*********************************************************')


def main():
    house = 1
    rock = 0
    attr = 1
    log_result(house=house,
               rock=rock,
               attr=attr)

    res = go(house, rock, attr)
    if res == 1:
        print("Ты мне нравишься")
    else:
        print("Созвонимся")


if __name__ == '__main__':
    main()