from sklearn import svm
from TrainEqCheck import learnerTest


model = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', 
        fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=1, max_iter=1000)
learnerTest.func_main(model)