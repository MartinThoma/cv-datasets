#!/usr/bin/env python


"""Analyze a cifar100 keras model."""

from keras.models import load_model
from keras.datasets import cifar100
from sklearn.model_selection import train_test_split
import numpy as np
import json
import io
import matplotlib.pyplot as plt
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

n_classes = 100


def plot_cm(cm, zero_diagonal=False):
    """Plot a confusion matrix."""
    n = len(cm)
    size = int(n / 4.)
    fig = plt.figure(figsize=(size, size), dpi=80, )
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(cm), cmap=plt.cm.viridis,
                    interpolation='nearest')
    width, height = cm.shape
    fig.colorbar(res)
    plt.savefig('cifar100-confusion-matrix.png', format='png')

# Load model
model = load_model('cifar100-2.h5')
print(model.summary())
from keras.utils.visualize_util import plot
plot(model, to_file='cifar100-model.png')

# Load validation data
(X, y), (X_test, y_test) = cifar100.load_data()

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.20,
                                                  random_state=42)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_val /= 255
X_test /= 255
from keras import backend as K
print("image_dim_ordering: %s" % K.image_dim_ordering())
print("shape=%s" % str(X_train.shape))

# Calculate confusion matrix
y_val_i = y_val.flatten()
y_val_pred = model.predict(X_val)
y_val_pred_i = y_val_pred.argmax(1)
cm = np.zeros((n_classes, n_classes), dtype=np.int)
for i, j in zip(y_val_i, y_val_pred_i):
    cm[i][j] += 1

acc = sum([cm[i][i] for i in range(100)]) / float(cm.sum())
print("Accuracy: {:0.1f}%".format(acc * 100))

# Create plot
plot_cm(cm)

# Serialize confusion matrix
with io.open('cm.json', 'w', encoding='utf8') as outfile:
    str_ = json.dumps(cm.tolist(),
                      indent=4, sort_keys=True,
                      separators=(',', ':'), ensure_ascii=False)
    outfile.write(to_unicode(str_))
