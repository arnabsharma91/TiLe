
from sklearn.tree import DecisionTreeClassifier
from TrainEqCheck import learnerTest

model = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=10, min_samples_split=2, 
                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=3, random_state=1, 
                         max_leaf_nodes=20, min_impurity_decrease=0.0, min_impurity_split=None, class_weight="balanced", 
                               presort=False)

learnerTest.func_main(model)
   

