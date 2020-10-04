# TiLe
![alt text](https://github.com/arnabsharma91/TiLe/blob/master/TiLeLogo.JPG)
Testing of learning algorithms.
This repositiry contains the code to test a given Machine Learning algorithm for **balancedness** property. Given a machine learning algorithm, our tool can check whether it is using the training data in a balanced way.
Till now, our testing framework is only limited to __scikit-learn__ library. Our testing framework is devloped based on the idea of metamorphic testing. You can learn more about our work by having a look at our [paper](https://ieeexplore.ieee.org/abstract/document/8730187). 

# Contributors
[Arnab Sharma](https://en.cs.uni-paderborn.de/sms/team/people/arnab-sharma), [Heike Wehrheim](https://en.cs.uni-paderborn.de/sms/team/people/heike-wehrheim)

# Requirements
We have developed this tool in Python3.6. To use our tool you need to install Python3 or higher. You will also need to install scipy, pandas, numpy, scikit-learn. All these required packages can be installed by using the `pip`command. We also use an SMT solver z3 in our framework. You can also install it by using `pip`command.
Example:<br>
```pip install <package-name>```

# Usage of TiLe
In the beginning you have to set some parameters for our tool to run. There are essentially 4 parameters which you need to fix: TRAIN_RATIO, MAX_INPUT, ROW_PERM_PER, COL_PERM_PER.
The first parameter indicates how much percentage of training data should be used as test data. MAX_INPUT indicates number of test data needs to be randomly generated. The third and the fourth parameters indicate the number of row and column permutation respectively, you would want the testing tool to consider. After fixing those paramters you are ready to run the tool.<br>
To run our tool you have to first `import` the `learnerTest.py` file <br> 
```from TrainEqCheck import learnerTest``` <br>
Then simply pass your classifier after fixing the hypeparameters through following: <br>
```learnerTest.func_main(model)``` <br>
Assuming `model`contain your classifier. We have added some sample test files which you can directly use. Example usage:<br>
```python DecisionTreeTest.py```<br>
At the end *Tile* will print whether or not the given learning algorithm is balanced or not.

