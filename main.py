import sys
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

from classifier.NNClassifier import NNClassifier


def main(fpath, pics_path):
    data = np.loadtxt(fpath, delimiter=',', dtype=int)
    data = np.array(data)
    total_rows = data.shape[0]
    train_rows = int(total_rows*.8)
    test_rows = int(total_rows*.2)
    [x_train, y_train, x_test, y_test] = \
        verify_representativity(data, total_rows, train_rows, test_rows)
    ann = NNClassifier(x_train, y_train, x_test, y_test)

    print('Item 3, secciones a, b, c')
    # 3 a,b,c
    ann.fit_model(layer_sizes=[40])
    ann.predict(ann.x_val, ann.y_val,
                cm_plot_name=pics_path+'/one_layer_val.png')

    print('Item 4, seccion a')
    print('\nPrueba con una capa de 10 neuronas (sobre conjunto de validacion)')
    ann.fit_model(layer_sizes=[10])
    ann.predict(ann.x_val, ann.y_val,
                cm_plot_name=pics_path+'/neurons10.png')

    print('\nPrueba con una capa de 32 neuronas (sobre conjunto de validacion)')
    ann.fit_model(layer_sizes=[32])
    ann.predict(ann.x_val, ann.y_val,
                cm_plot_name=pics_path + '/neurons32.png')

    print('\nPrueba con una capa de 64 neuronas (sobre conjunto de validacion)')
    ann.fit_model(layer_sizes=[64])
    ann.predict(ann.x_val, ann.y_val,
                cm_plot_name=pics_path + '/neurons64.png')

    print('Item 4, seccion b')
    print('\nPrueba con una capa de 64 neuronas, usando funcion sigmoide\n'
          '(sobre conjunto de prueba)')
    ann.fit_model(layer_sizes=[64], n_train_times=3)
    ann.predict(ann.x_test, ann.y_test, cmap=plt.cm.Reds,
                cm_plot_name=pics_path+'/opt_test_sig.png')

    print('\nPrueba con una capa de 64 neuronas, usando funcion ReLu\n'
          '(sobre conjunto de prueba)')
    ann.fit_model(layer_sizes=[64], active_func='relu', n_train_times=3)
    ann.predict(ann.x_test, ann.y_test, cmap=plt.cm.Reds,
                cm_plot_name=pics_path+'/opt_test_relu.png')


def everything_is_ok(rates, train_rates, test_rates):

    is_ok = True
    for i in np.arange(0, 10):
        if not ((rates[i] - .05) <= train_rates[i] <= (rates[i] + .05) and
                (rates[i] - .05) <= test_rates[i] <= (rates[i] + .05)):
            is_ok = False
            break
    return is_ok


def verify_representativity(data, total_rows, train_rows, test_rows):
    rates = {}
    for i in np.arange(0, 10):
        rates[i] = float((data[: total_rows, 64] == i).sum())/float(total_rows)

    shuffle(data)
    train_rates = {}
    test_rates = {}

    while True:
        x_train, y_train = data[:train_rows, :64], data[:train_rows, 64]
        x_test, y_test = data[train_rows:total_rows, :64], data[train_rows:total_rows, 64]

        for i in np.arange(0, 10):
            train_rates[i] = float((y_train == i).sum())/train_rows
            test_rates[i] = float((y_test == i).sum())/test_rows

        if everything_is_ok(rates, train_rates, test_rates):
            break

    return [x_train, y_train, x_test, y_test]


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])