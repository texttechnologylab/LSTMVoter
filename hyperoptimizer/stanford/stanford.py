import subprocess
import os
from subprocess import check_output
import shlex
import re
from shutil import copyfile
from hyperopt import STATUS_OK
import json
from shutil import copyfile


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
    
    stanfordJar = os.path.abspath("lib/stanford-ner.jar")
    args = "java -Xmx150g -cp " + stanfordJar + " edu.stanford.nlp.ie.crf.CRFClassifier  -serializeTo ner-model.ser.gz "+params
    run_command(args)

def test(parametersTest,parametersTrain):
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
    
    stanfordJar = os.path.abspath("lib/stanford-ner.jar")
    out = run_command("/usr/bin/java -Xmx150g -cp '" + stanfordJar + "' edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner-model.ser.gz "+parametersTest)
    print "/usr/bin/java -Xmx150g -cp '" + stanfordJar + "' edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner-model.ser.gz "+parametersTest
    if(not os.path.exists("models")):
        os.mkdir("models")

    print out
    entity, p, r, f1, tp, fp, fn =  out.split("\t")
    copyfile("ner-model.ser.gz","models/ner-model.ser.gz." + str(f1))
    
    with open("models/ner-model.ser.gz.config."+str(f1), "w") as text_file:
        text_file.write(parametersTrain)

    print "============================="
    print entity, p, r, f1, tp, fp, fn
    print "============================="
    print 
    print 
    print
    return float(f1.replace(",","."))
    
def f(params):
    print params
    print json.dumps(params, indent=4, sort_keys=True)
    parametersTrain = "";
    parametersTest = "";
    for i in params:
        if i == "gazetteOptions":
            if params[i]["useGazettes"] == "true":
                for j in params[i]: 
                    print j
                    if j == "sloppyGazette":
                        if str(params[i][j]) == str("true"):
                            parametersTrain += " -"+j+" '" +str(params[i][j])+"'"
                        else:
                            parametersTrain += " -cleanGazette 'true'"
                    else:
                        parametersTrain += " -"+j+" '" +str(params[i][j])+"'"
        elif i == "types":
            print i
        elif i == "testFile":
            parametersTest = " -testFile "+str(params[i])
        else:
            parametersTrain += " -"+i+" '" +str(params[i])+"'"
    print "training with: ",parametersTrain
    print "test with: ",parametersTest
    train(parametersTrain)
    print "test"
    acc = test(parametersTest,parametersTrain)

    print "f1:", acc
    return {'loss': -acc, 'status': STATUS_OK}
