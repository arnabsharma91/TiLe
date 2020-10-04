from sklearn.naive_bayes import GaussianNB
from TrainEqCheck import learnerTest

model = GaussianNB()
learnerTest.func_main(model)