from sklearn.ensemble import AdaBoostClassifier
from TrainEqCheck import learnerTest

model = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', 
                           random_state=1)
                           
learnerTest.func_main(model)