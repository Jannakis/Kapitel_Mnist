import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.ndimage.interpolation import shift

def shift_image(image, direction):
    return shift(image.reshape((28, 28)), direction, cval=0).reshape(784)

def augment_mnist_dataset(X_train, y_train):
    X_augmented = []
    y_augmented = []
    
    for image, label in zip(X_train, y_train):
        X_augmented.append(image)
        y_augmented.append(label)
        
        # Create shifted copies
        X_augmented.append(shift_image(image, [1, 0]))  # Down
        X_augmented.append(shift_image(image, [-1, 0]))  # Up
        X_augmented.append(shift_image(image, [0, 1]))  # Right
        X_augmented.append(shift_image(image, [0, -1]))  # Left
        
        y_augmented.extend([label] * 4)
    
    return np.array(X_augmented), np.array(y_augmented)

# Import Data
test_data = pd.read_csv(r"A:\Desktop\Uni\WHK_Jelali\Mnist\mnist_test.csv")
train_data = pd.read_csv(r"A:\Desktop\Uni\WHK_Jelali\Mnist\mnist_train.csv")

# split data
X_train = train_data.iloc[:,1:].to_numpy()
y_train =  train_data.iloc[:,0].to_numpy()
X_test = test_data.iloc[:,1:].to_numpy()
y_test = test_data.iloc[:,0].to_numpy()

# Modell importieren
knn_clf = KNeighborsClassifier()

# Parameter-Grid definieren
#param_grid = {    'n_neighbors':[1,2]}
#grid_search = GridSearchCV(knn_clf, param_grid, scoring='accuracy')
#grid_search.fit(X_train, y_train)

knn_clf.fit(X_train,y_train)

acc = accuracy_score(y_test,knn_clf.predict(X_test))
print(f"Korrektklassifikation normaler Datensatz: {acc}")

# Datensatz variiren durch Imageshift
X_train_augmented, y_train_augmented = augment_mnist_dataset(X_train, y_train)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_augmented,y_train_augmented)

acc_ag = accuracy_score(y_test,knn_clf.predict(X_test))
print(f"Korrektklassifikation bearbeiteter Datensatz: {acc_ag}")




