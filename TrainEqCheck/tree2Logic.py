
import pandas as pd
import csv as cv
import sys
from sklearn import tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import fileinput
import re
import os


from sklearn.tree import _tree
def tree_to_code(tree, feature_names, turn):
    
    if(turn == 0):
        f = open('TreeOutputBefore.txt', 'w')
    else:
        f = open('TreeOutputAfter.txt', 'w')
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print("def tree({}):".format(", ".join(feature_names)))
    f.write("def tree({}):".format(", ".join(feature_names)))
    f.write("\n")
    

    def recurse(node, depth):
        indent = "  " * depth
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            #print("{}if {} <= {}:".format(indent, name, threshold))
            f.write("{}if {} <= {}:".format(indent, name, threshold))
            f.write("\n")
            
            #print("{}".format(indent)+"{")
            f.write("{}".format(indent)+"{")
            f.write("\n")
            
            recurse(tree_.children_left[node], depth + 1)
            
            #print("{}".format(indent)+"}")
            f.write("{}".format(indent)+"}")
            f.write("\n")
            
            
            #print("{}else:  # if {} > {}".format(indent, name, threshold))
            f.write("{}else:  # if {} > {}".format(indent, name, threshold))
            f.write("\n")
            
            #print("{}".format(indent)+"{")
            f.write("{}".format(indent)+"{")
            f.write("\n")
            
            recurse(tree_.children_right[node], depth + 1)
            
            #print("{}".format(indent)+"}")
            f.write("{}".format(indent)+"}")
            f.write("\n")
            
        else:
            #print("{}return {}".format(indent, np.argmax(tree_.value[node][0])))
            f.write("{}return {}".format(indent, np.argmax(tree_.value[node][0])))
            f.write("\n")
            #print("{}".format(indent)+"}")
            
    
    recurse(0, 1)
    f.close() 

def file_len(fname):
    #i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def funcConvBranch(single_branch, dfT, rep):
    f3 = open('DecSmt.smt2', 'a') 
    f3.write("(=> (and ")
    for i in range(0, len(single_branch)):
        temp_Str = single_branch[i]
        if('if' in temp_Str):
            for j in range (0, dfT.columns.values.shape[0]):
                if(dfT.columns.values[j] in temp_Str):
                    fe_name = str(dfT.columns.values[j])
                    fe_index = j
                    
            data_type = str(dfT.dtypes[fe_index])
            
            if('<=' in temp_Str):
                sign = '<='
            elif('<=' in temp_Str):
                sign = '>'    
            elif('>' in temp_Str):
                sign = '>'
            elif('>=' in temp_Str):
                sign = '>='

            num_data = temp_Str.split(sign)[1]
            digit = float(re.search(r'\d+.\d+', num_data).group(0))
            digit = str(digit) 
            f3.write("(" + sign + " "+ fe_name +" " + digit +") ")
                
        elif('return' in temp_Str):
            digit_class = int(re.search(r'\d+', temp_Str).group(0))
            digit_class = str(digit_class)
            f3.write(") (= Class " +digit_class +"))")
            f3.write('\n')
    f3.close()
    

def funcGetBranch(sinBranch, dfT, rep):
    flg = False
    for i in range (0, len(sinBranch)):
        tempSt = sinBranch[i]
        if('return' in tempSt):
            flg = True
            funcConvBranch(sinBranch, dfT, rep)
            

def funcGenBranch(dfT, turn):
    f3 = open('DecSmt.smt2', 'a') 
    f3.write('\n \n')
    if(turn == 0):
        file = 'TreeOutputBefore.txt'
        f3.write(';Encoding for the tree before \n')
        f3.write('(assert (= p (and \n')
    else:
        f3.write(';Encoding for the tree after \n')
        f3.write('(assert (= q (and \n')
        file = 'TreeOutputAfter.txt'
    f3.close()    
    with open(file) as f1:
        file_content = f1.readlines()
    file_content = [x.strip() for x in file_content] 
    f1.close()
    noOfLines = file_len(file)
    temp_file_cont = ["" for x in range(noOfLines)]
    
    i = 1
    k = 0
    while(i < noOfLines):
        j = k-1
        if(temp_file_cont[j] == '}'):
            funcGetBranch(temp_file_cont, dfT, turn)
            while(True):
                if(temp_file_cont[j] == '{'):
                    temp_file_cont[j] = ''
                    temp_file_cont[j-1] = ''
                    j = j-1
                    break  
                elif(j>=0):    
                    temp_file_cont[j] = ''
                    j = j-1
            k = j    
        else:    
            temp_file_cont[k] = file_content[i]
            k = k+1
            i = i+1

    if('return' in file_content[1]):
        digit = int(re.search(r'\d+', file_content[1]).group(0))
        f3 = open('DecSmt.smt2', 'a') 
        if(turn == 0):
            f3.write("(assert (= Class "+str(digit)+"))")
        f3.write("\n")
        f3.close()
    else:    
        funcGetBranch(temp_file_cont, dfT, turn)

        
def funcConv(dfT):
    temp_content = ['']
    turn = 0
    min_val = 0
    max_val = 0 
    #Declaring variables first time
    f2 = open('DecSmt.smt2', 'w')
    for i in range (0, dfT.columns.values.shape[0]):
        min_val = dfT.iloc[:, i].min()
        max_val = dfT.iloc[:, i].max()
        tempStr = dfT.columns.values[i]
        fe_type = dfT.dtypes[i]
        fe_type = str(fe_type)
        if 'int' in fe_type:
            f2.write("(declare-fun " + tempStr+ " () Int) \n")
            f2.write("(assert (and (>= "+tempStr+" "+str(min_val)+")"+" "+"(<= "+tempStr+" "+str(max_val)+"))) \n")
        elif 'float' in fe_type:
            f2.write("(declare-fun " + tempStr+ " () Real) \n")
            f2.write("(assert (and (>= "+tempStr+" "+str(round(min_val, 2))+")"+" "+"(<= "+tempStr+" "+str(round(max_val, 2))+"))) \n")
    f2.write("; Variables for the tree_before \n")
    f2.write("(declare-fun p () Bool)\n")
    f2.write("; Variables for the tree_after \n")
    f2.write("(declare-fun q () Bool)\n")
    f2.write('\n')
    f2.close()
  
    #Calling function to get the branch and convert it to z3 form
    funcGenBranch(dfT, 0)
    f3 = open('DecSmt.smt2', 'a')
    f3.write(')))')
    f3.close()
    #Calling function to get the logical formula for the tree_after
    funcGenBranch(dfT, 1)
    f3 = open('DecSmt.smt2', 'a')
    f3.write(')))')
    f3.close()
    
    
def funcGenSMT(df):
    funcConv(df)
    f = open('DecSmt.smt2', 'a') 
    f.write("\n \n")
    f.write('(assert (or (and p (not q)) (and q (not p))))\n')
    f.write("(check-sat) \n")
    f.close()
    os.system(r"z3 DecSmt.smt2 > FinalOutput.txt")


def functree2LogicMain(tree_before, tree_after, df):
    tree_to_code(tree_before, df.columns, 0)
    tree_to_code(tree_after, df.columns, 1)
    #Generating SMT file for Z3 and adding non monotonicity constraint
    funcGenSMT(df)
    with open('FinalOutput.txt') as f1:
        file_content = f1.readlines()
    file_content = [x.strip() for x in file_content]
    result = file_content[0]
    if result == 'sat':
        return True
    return False
