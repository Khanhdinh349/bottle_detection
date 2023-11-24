import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn import svm
import joblib
# from sklearn.externals import joblib

def data(X_path,Y_path):
  """Open .pickle files, and restore X and Y lists"""
  pickle_in = open(X_path, "rb")
  X = pickle.load(pickle_in)
  pickle_in = open(Y_path, "rb")
  Y = pickle.load(pickle_in)
  return X, Y

def sift(img):
  """Create SIFT method to exclude features, and return kp and des"""
  sift = cv2.xfeatures2d.SIFT_create()
  kp, des = sift.detectAndCompute(img,None)
  return kp, des

def orb(img):
  """Create ORB method to exclude features, and return kp and des"""
  orb = cv2.ORB_create()
  kp, des = orb.detectAndCompute(img,None)
  return kp, des

def surf(img):
  """Create SURF method to exclude features, and return kp and des"""
  surf = cv2.xfeatures2d.SURF_create()
  kp, des = surf.detectAndCompute(img,None)
  return kp, des

def feature_number(feature):
  """Creating a list with the features of individual images, and returning list_data and ind"""
  ind = []
  list_data = []
  t0 = time()
  for i in range(len(X)):
    kp, des = feature(X[i])
    if len(kp) < 20:
      ind.append(i)
      continue
    des = des[0:20,:]
    vector_data = des.reshape(1,len(des)*len(des[1]))
    list_data.append(vector_data)
  features = ['sift', 'surf', 'orb']
  print("Algorithm time: %0.3fs" % (time() - t0))
  return list_data, ind
    
def svm_parameters(X_train, y_train):
  """Finding parameters for model training and returning clf.best_estimator_"""
  t0 = time()
  param_grid = {'C': [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.001, 0.01, 0.1], 
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
  clf = GridSearchCV(
    svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
  clf = clf.fit(X_train, y_train)
  print("Parameter finding time: %0.3fs" % (time() - t0))
  return clf.best_estimator_

def svm_train(X_train, y_train):
  """Model training and returning clf"""
  t0 = time()
  clf = svm.SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1e-8, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
  clf.fit(X_train, y_train)
  print("Model training time: %0.3fs" % (time() - t0))
  return clf

def svm_test(clf, X_test, y_test):
  """Testing the model and returning y_pred"""
  t0 = time()
  y_pred = clf.predict(X_test)
  print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
  print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))
  print("Recall:",metrics.recall_score(y_test, y_pred, average='micro'))
  print(classification_report(y_test, y_pred, target_names=categories))
  print("Model testing time: %0.3fs" % (time() - t0))
  return y_pred

def svm_save(clf, path):
  """Saving SVM model"""
  joblib.dump(clf, path)

def plot_gallery(images, titles, h, w, n_row=1, n_col=2):
  """Displays individual images, image categories, and default categories"""
  plt.figure(figsize=(4 * n_col, 2 * n_row))
  plt.subplots_adjust(bottom=0, left=0.1, right=0.9, top=.95, hspace=.35)
  for i in range(n_row * n_col):
    plt.subplot(n_row, n_col, i + 1)
    plt.imshow(images[i].reshape((w,h)))
    plt.title(titles[i], size=10)
    plt.xticks(())
    plt.yticks(())

def title(y_pred, y_test, target_names, i):
  """Extract the actual and default image categories, and return pred_name and true_name"""
  pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
  true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
  return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

if __name__ == '__main__': 
  categories = ["Glass", "Plastic", "Cans"]
  X_path = "D:\\Start-up\\ATM\\pickle\\x.pickle"
  Y_path = "D:\\Start-up\\ATM\\pickle\\y.pickle"
  IMG_W = int(640/2)
  IMG_H = int(480/2)
  X, Y = data(X_path, Y_path)
  features = ['sift', 'surf', 'orb']
  a = 0
  for feature in [sift, surf, orb]:
    t1 = time()
    labels = Y[:]
    list_data, ind = feature_number(feature)
    for i in sorted(ind, reverse=True):
      del labels[i]
    data = np.array(list_data).reshape(len(labels),len(list_data[0][0]))
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=42) # 70% training and 30% test
    #clf = svm_parameters(X_train, y_train)
    clf = svm_train(X_train, y_train)
    y_pred = svm_test(clf, X_test, y_test)
    print("The time of the whole program: %0.3fs" % (time() - t1))
    save_path = "D:\\Start-up\\ATM\\" + str(features[a]) + "_trained_model.npy"
    svm_save(clf, save_path)
    a += 1
    prediction_titles = [title(y_pred, y_test, categories, i)
                        for i in range(y_pred.shape[0])]
    #plot_gallery(X_test, prediction_titles, 80,80)
    #plot_gallery(X_test, prediction_titles, 50,64)
    #plot_gallery(X_test, prediction_titles, 80,20)
    plt.show()
  