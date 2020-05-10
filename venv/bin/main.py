import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score


# Step One: Load the Data

path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cell_samples.csv"

df = pd.read_csv(path)

print(df.head())

# Class 2 is benign, class 4 is malignant
axis = df[df["Class"] == 4].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant')
df[df["Class"]==2].plot(kind="scatter", x="Clump", y="UnifSize", color="DarkBlue", label="benign")
plt.show()

# Seems like, at first glance, malignant bumps tend to have higher clump and size values.
# However, this is intuition, and a weak one at that.

print(df.dtypes)

# Step 2: Preprocessing
# We need features as a number if it will be used in SVM. We can drop the non-numerical rows in
# BareNuc
df["BareNuc"] = df["BareNuc"].apply(pd.to_numeric, errors="coerce")
df.dropna(axis=0, subset=["BareNuc"], inplace=True)
df["BareNuc"] = df["BareNuc"].astype("int")

print(df.dtypes)
print(df["BareNuc"].head(20))

# Step 3: Now that the data is processed, we need to get our data sets

X = np.asarray(df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
y = np.asarray(df[["Class"]])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=4)

# Running SVM
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)

print(yhat[0:5])

# Evaluation

# Confusion Matrix

# Function Copied from IBM
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Compute Confusion matrix
matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot the matrix
plt.figure()
plot_confusion_matrix(matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

# WOW this seems pretty great

# F1 Score
print(f1_score(y_test, yhat, average='weighted') )

# Jaccard Index
print(jaccard_similarity_score(y_test, yhat))

# Quite good. This is a solid model!


