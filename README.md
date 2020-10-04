# TiLe
Testing of learning algorithms.
This repositiry contains the code to test a given Machine Learning algorithm for **balancedness** property. Given a machine learning algorithm, our tool can check whether it is using the training data in a balanced way.
Till now, our testing framework is only limited to __scikit-learn__ library. Our testing framework is devloped based on the idea of metamorphic testing. You can learn more about our work by having a look at our [paper](https://ieeexplore.ieee.org/abstract/document/8730187). 
# Contributors
[Arnab Sharma](https://en.cs.uni-paderborn.de/sms/team/people/arnab-sharma), [Heike Wehrheim](https://en.cs.uni-paderborn.de/sms/team/people/heike-wehrheim)
# Usage of TiLe
In the beginning you have to set some parameters for our tool to run. 

You have to first `<import>` the `<learnerTest.py>` file 
```from TrainEqCheck import learnerTest```
Then simply pass your classifier after fixing the hypeparameters through following:
```learnerTest.func_main(model)```
Assuming `<model>`contain your classifier. We have added some sample test files which you can directly use. 
