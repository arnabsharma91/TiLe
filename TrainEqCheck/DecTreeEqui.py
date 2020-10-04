
from TrainEqCheck import tree2Logic
from joblib import load
import os


def func_main(df, meta_op):
    model_before = load('model_before.joblib')
    model_after = load('model_after.joblib')
    z3_result = tree2Logic.functree2LogicMain(model_before, model_after, df)
    os.remove('DecSmt.smt2')
    os.remove('TreeOutputBefore.txt')
    os.remove('TreeOutputAfter.txt')
    os.remove('FinalOutput.txt')
    if z3_result:
        print('The given Decision tree model is not balanced under '+meta_op)
        return True   
    return False

