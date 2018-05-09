import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


class NNClassifier(object):

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.defined = False
        self.x_val = None
        self.y_val = None
        self.clf = None

    def train_eval_validation(self, layer_sizes, active_func='logistic', n_train_times=1):

        self.clf = MLPClassifier(activation=active_func, hidden_layer_sizes=layer_sizes,
                                 solver='adam', early_stopping=True,
                                 validation_fraction=0.2)

        if not self.defined:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, np.ravel(self.y_train),
                                              random_state=self.clf.random_state,
                                              test_size=self.clf.validation_fraction)
            self.defined = True
        else:
            train_test_split(self.x_train, np.ravel(self.y_train),
                             random_state=self.clf.random_state,
                             test_size=self.clf.validation_fraction)

        for i in np.arange(n_train_times):
            print('Training %d' % (i+1))
            self.clf.fit(self.x_train, np.ravel(self.y_train))

        self.clf.out_activation_ = 'softmax'

    def predict(self, x_, y_, cm_plot_name):
        y_pred = self.clf.predict(x_)
        cm = confusion_matrix(y_, y_pred)
        plot_confusion_matrix(cm, cm_plot_name)


def plot_confusion_matrix(cm, cm_plot_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues
                          ):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else '{:d}'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, fmt.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(cm_plot_name)
    plt.clf()