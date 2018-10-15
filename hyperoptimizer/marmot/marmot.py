import subprocess
import os
from subprocess import check_output
import shlex
import re
from shutil import copyfile
from hyperopt import STATUS_OK
import json
import conlleval
from cStringIO import StringIO
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def train(params):
    def run_command(command):
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print output.strip()
        rc = process.poll()
    marmotJar = os.path.abspath("lib/marmot-2015-10-22.jar")

    args = "java -Xmx5G -cp "+marmotJar+" marmot.morph.cmd.Trainer -tag-morph false -model-file en.marmot "+params
    print args
    run_command(args)

def test(parametersTest):
    def run_command(command):
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        f1 = None
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if bool(re.search('Totals',output.strip())):
                f1 = output.strip()
        rc = process.poll()
        return f1
    marmotJar = os.path.abspath("lib/marmot-2015-10-22.jar")
    
    out = run_command("java -cp " + marmotJar + " marmot.morph.cmd.Annotator --model-file en.marmot --pred-file text.out.txt "+parametersTest)
    
    conll = addAdditionalAnnotationFileToFile("/home/ahemati/workspaceGit/LSTMVoter/data/CHEMDNER/first_step_train/training.small.conll.txt","text.out.txt", 5)
    output =  toString(conll)
    precision, recall, f1 = conlleval.global_evaluate(output)
    print precision, recall, f1 
    return f1

def addAdditionalAnnotationFileToFile(inputFile, newFile,annotationIndex=-1):
    f = open(inputFile)
    output = []
    f_newFile = open(newFile)
    for l in f_newFile:
        split = l.strip().split("\t");
        if(len(split)>1):
            output.append(l.strip().split("\t")[annotationIndex].strip())
        else:
            output.append(l.strip())
        
    combined = conllToAdvanced(inputFile, output);
    return combined

def conllToAdvanced(inputFile,output):
    f = open(inputFile)
    combined = []
    for i,l in enumerate(f):
        if(len(l.strip())>0):
            combined.append(l.strip()+"\t"+output[i])
    return combined

def toString(input):
    output = StringIO()
    for i in input:
        output.write(i)
        output.write("\n")
    return output.getvalue()
    
def f(params):
    print params
    print json.dumps(params, indent=4, sort_keys=True)
    parametersTrain = "";
    parametersTest = "";
    for i in params:
        if i == "testFile":
            parametersTest = " --test-file form-index=0,"+str(params[i])
        elif i == "trainFile":
            parametersTrain += " -train-file form-index=0,tag-index=1,"+params[i]
        else:
            parametersTrain += " -"+i+" '" +str(params[i])+"'"
    print "training with: ",parametersTrain
    print "test with: ",parametersTest
    train(parametersTrain)
    print "test"
    acc = test(parametersTest)

    print "f1:", acc
    return {'loss': -acc, 'status': STATUS_OK}
