from hyperopt import Trials
from hyperopt.mongoexp import MongoTrials
import os

trials = Trials() 
from io import StringIO

import json
from pprint import pprint

def saveStringToFile(string, outputFile):
    with open(outputFile, "w") as text_file:
        text_file.write(string)

def readAdvancedFile(inputFilePath):
    file = open(inputFilePath, "r")
    output = []
    for line in file:
        output.append(line)
    return output;

def advancedToBioCreative(input,annotationIndex=-1):
    annotations = [];
    previousTag = [];
    isInTag = False;
    beginTag = None;
    currentTag = "";
    
    printAbHier = False;
    for string in input:
        split = string.strip().split("\t")
        if len(split)==1:
            if isInTag:
                annotations.append(previousTag[4]+"\t"+previousTag[5]+"\t"+beginTag[2]+"\t"+previousTag[3]+"\t"+currentTag+"\t"+beginTag[1].replace("B-", ""))
            isInTag = False
            currentTag = ""
        if split[annotationIndex].startswith("B"):
            if isInTag:
                annotations.append(previousTag[4]+"\t"+previousTag[5]+"\t"+beginTag[2]+"\t"+previousTag[3]+"\t"+currentTag+"\t"+beginTag[1].replace("B-", ""))
                currentTag = ""
            isInTag = True
            beginTag = split
        elif split[annotationIndex].startswith("O"):
            if isInTag:
                annotations.append(previousTag[4]+"\t"+previousTag[5]+"\t"+beginTag[2]+"\t"+previousTag[3]+"\t"+currentTag+"\t"+beginTag[1].replace("B-", ""))
            isInTag = False
            currentTag = ""
        if isInTag:
            if(len(currentTag)!=0):
                for i in range(int(split[2])-int(previousTag[3])):
                    currentTag+=" "
            currentTag += split[0]
        previousTag = split
    return annotations 
    
def biocreativeToSimpleBioCreative(input,toString=False):
    output = []
    for i in input:
        split = i.split("\t")
        output.append(split[0]+"\t"+split[1]+":"+split[2]+":"+split[3])
    
    if(toString):
        outputString = ""
        for j in output:
            outputString = outputString + j +"\n"
        return outputString
    else:
        return output
    
def conllToAdvanced(inputFile,output):
    f = open(inputFile)
    combined = []
    for i,l in enumerate(f):
        combined.append(l.strip()+"\t"+output[i])
    return combined

def listToFile(input,file):
    with open(file, 'w') as f:
        for item in input:
            f.write("%s\n" % item)

def advancedToConll(inputFile,asString=False,wordIndex=0,tagIndex=1):
    f = open(inputFile)
    combined = []
    for l in f:
        split = l.strip().split()
        if(len(split)>0):
            combined.append([split[wordIndex],split[tagIndex]])
        else:
            combined.append([])
    if(asString):
        return toString(combined)
    else:
        return combined

def toString(input):
    output = StringIO()
    for i in input:
        for j in i:
            output.write(j)
            output.write("\t")
        output.write("\n")
    return output.getvalue()


def addAdditionalAnnotationFileToFile(inputFile, newFile,annotationIndex=-1):
    f = open(inputFile)
    output = []
    f_newFile = open(newFile)
    for l in f_newFile:
        if(l.strip().startswith("#")):
            continue
        split = l.strip().split();
        if(len(split)>1):
            output.append(l.strip().split()[annotationIndex].strip())
        else:
            output.append(l.strip())
    
    combined = conllToAdvanced(inputFile, output);

    return combined

def addAdditionalAnnotationToFile(inputFile, output,asString=False):
    f = open(inputFile)
    combined = []
    enumaer = enumerate(f)
    for i,l in enumaer:
        combined.append((l.strip()+"\t"+output[i]).split("\t"))
        
    if(asString):
        return toString(combined)
    else:
        return combined
    
def addAditionalAnnotations(inputFile, additionalFeatures,asString=False):
    f = open(inputFile)
    combined = []
    enumaer = enumerate(f)
    for i,l in enumaer:
        line = l.strip()
        for feature in additionalFeatures:
            line = line + "\t" + feature[i]
        combined.append(line.split("\t"))
        
    if(asString):
        return toString(combined)
    else:
        return combined

def readOutputFile(file,outputIndex):
    f = open(file)
    output = []
    for l in f:
        split = l.strip().split("\t");
        if(len(split)>1):
            output.append(split[outputIndex])
        else:
            output.append("")
    return output
            
def readSentences(file,wordIndex = 0):
    f = open(file)
    output = []
    tmp = []
    for l in f:
        split = l.strip().split("\t");
        if(len(split)>1):
            tmp.append(split[wordIndex])
        else:
            output.append(tmp)
            tmp = []
    return output

def loadJsonParams(paramsFile):
    import re
    s = "Example String"
    replaced = re.sub(r'(.*?)St.*', r'\1', s)
    with open(paramsFile) as myfile:
        data=myfile.read().replace('\n', '').replace("'","\"")
        replaced = re.sub(r'.*(.vals.:.*?}).*', r'{\1}', data)
        d = json.loads(replaced)
        return d['vals']


def createFile(baseFile):
    file = baseFile
    
    withMarmot = addAdditionalAnnotationFileToFile(file, file+ ".marmot", 2)
    listToFile(withMarmot, file + "_with_Marmot.txt")
    
    withAnago = addAdditionalAnnotationFileToFile(file + "_with_Marmot.txt", file + ".anago", 2)
    listToFile(withAnago, file + "_with_Marmot_Anago.txt")
    
#     withUDPipe = addAdditionalAnnotationFileToFile(file + "_with_Marmot_Anago.txt", file + ".udpipe", 3)
#     listToFile(withUDPipe, file + "_with_Marmot_Anago_UDPipe.txt")
    
    os.remove(file + "_with_Marmot.txt")


def conllToConllU(inputFile):
    f = open(inputFile)
    data = []
    output = StringIO()
    for l in f:
        split = l.strip().split("\t");
        if(len(split)>1):
            data.append((split[0],split[-1]))
        else:
            data.append("")
    
    count = 1
    for i in data:
        if(len(i) == 0):
            output.write("\n")
            count = 1
        else:
            output.write("{}\t{}\t_\t{}\t_\t_\t_\t_\t_\t_\n".format(count,i[0],i[1]))
            count = count + 1
    return output.getvalue()

if __name__ == "__main__":
    
#     goldStandardTrain = biocreativeToSimpleBioCreative(readAdvancedFile("/home/ahemati/Downloads/chemdner_corpus (2)/training.annotations.txt"),True)
#     with open("training.gold.annotations", "w") as text_file:
#         text_file.write(goldStandardTrain)

#     advancedData = readAdvancedFile("/home/staff_homes/ahemati/projects/biocreative/data/CHEMDNER/advanced/training.conll")
#     simpleBioCreative = biocreativeToSimpleBioCreative(advancedToBioCreative(advancedData,1), True);
#     with open("training.annotations", "w") as text_file:
#         text_file.write(simpleBioCreative)
    
#     marmotOutput = addAdditionalAnnotationToFile("/home/staff_homes/ahemati/projects/biocreative/data/CHEMDNER/first_step_train/evaluation.txt","/home/staff_homes/ahemati/projects/biocreative/hyperoptimizer/lib/marmot/text.out.txt",5)
#     marmotOutputSimpleBio = biocreativeToSimpleBioCreative(advancedToBioCreative(marmotOutput, -1), True);
#     print marmotOutputSimpleBio
#     with open("evaluation.txt", "w") as text_file:
#         text_file.write(marmotOutputSimpleBio)
    #pprint(loadJsonParams('models/hyperas/model.json.0.0'))
    
#     file = "merge_dev_voter"
#     
#     datadir = "/home/staff_homes/ahemati/projects/SequenceLabeling/data/LT4HALA/data_and_doc/split-40-10-40-10/"
#     withMarmot = addAdditionalAnnotationFileToFile(os.path.join(datadir,file + ".conllu.conll2002"), os.path.join(datadir,file + ".conllu.conll2002.marmot"), 2)
#     listToFile(withMarmot, file + "_with_Marmot.txt")
#     
#     withAnago = addAdditionalAnnotationFileToFile(file + "_with_Marmot.txt", os.path.join(datadir,file + ".conllu.conll2002.anago"), 2)
#     listToFile(withAnago, file + "_with_Marmot_Anago.txt")
#     
#     os.remove(file + "_with_Marmot.txt")

    print(conllToConllU("../../output.txt"))
