from sklearn.ensemble import BaggingClassifier
from TrainEqCheck import learnerTest

model = BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, 
                          bootstrap=True, bootstrap_features=False, oob_score=False, 
                          warm_start=False, n_jobs=1, random_state=1, verbose=0)
learnerTest.func_main(model)