import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
from cStringIO import StringIO


import numpy as np
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding, Input, concatenate, Reshape, Lambda, Concatenate, TimeDistributed, Subtract, Dot, Multiply
from keras.optimizers import RMSprop, Adam, Adamax
from keras import backend as K
from keras_contrib.layers import CRF
from keras.models import load_model
import conlleval
import json

from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras_attention_mechanism.attention_lstm import *

from utils import *

from argparse import ArgumentParser

import pickle

def checkFile(input_file):
    file = open(input_file, "r")
    prevLine = ""
    print "checking file:",input_file
    for i,line in enumerate(file):
        if len(prevLine.strip())==0 and len(line.strip())==0:
            raise Exception("Error in file. Two empty lines. Line:"+str(i))
        prevLine = line
        
def maxlen(input_file):
    file = open(input_file, "r")
    #preperation of input file. calculating max doc length and char_idx
    max_len = 0
    cur_len = 0
    for line in file:
        word = line.strip()
        #calculates the maximal sequence length. This is needed to add padding vectors to lstm.
        if(len(word) == 0):
            if(cur_len > max_len and cur_len > 0):
                max_len = cur_len
            cur_len = 0
        else:
            cur_len = cur_len + 1
    return max_len
        
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

def load_embeddings(dir,skipHeader=False):
    print "processing word embeddings file"
    print "start: counting lines"
    def file_len(fname):
        embedding_size = 0
        with open(fname) as f:
            for i, l in enumerate(f):
                if (not skipHeader and i == 0) or (skipHeader and i == 1):
                    embedding_size = len(l.split(" "))-1
                if i % 10001 == 0:
                    print "current embeddings count", i
                pass
            
        return i + 1, embedding_size
    
    size,embedding_size = file_len(dir)
    print "finish: counting lines. Words in embedding file: " , size
    
    embeddings_matrix = np.zeros((size+2, embedding_size))
    embeddings_index = {}
    f = open(dir)
    #embedding matrix starts with index 2. Index 0 is reserved for padding input and index 1 is reserved for unknown words.
    i = 2
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = i
        embeddings_matrix[i] = (coefs)
        i = i+1
        if(i%10000==0):
            progress(i,size,status="loading embeddings file")
#             break;
    progress(i,size,status="loading embeddings file")
    f.close()
    return embeddings_matrix,embeddings_index

def load_idxs(input_file,input_file_seperator,considerFeatures,max_len,idxs=None):
    file = open(input_file, "r")
    max_char_len = 20

    if idxs == None:
        idxs = {}
        idxs["char_idx"]={}
        idxs["output_idx"]={}
        if(considerFeatures!=None):
            for i in considerFeatures:
                idxs[i]={}
                
        #preperation of input file. calculating max doc length and char_idx
        cur_len = 0
        for line in file:
            split = line.split(input_file_seperator)
            word = split[0]
            word = word.strip()
            
            #calculates the maximal sequence length. This is needed to add padding vectors to lstm.
            if(len(word) == 0):
                if(cur_len > max_len and cur_len > 0):
                    max_len = cur_len
                cur_len = 0
            else:
                cur_len = cur_len +1
                #generates charindex for charembeddings
                for char in list(word):
                    if char not in idxs["char_idx"]:
                        idxs["char_idx"][char] = len(idxs["char_idx"]) + 1
                
                tmpOutput = split[1].strip()
                if tmpOutput not in idxs["output_idx"]:
                    idxs["output_idx"][tmpOutput] = len(idxs["output_idx"])
                    
                if considerFeatures !=None:
                    for i in considerFeatures:             
                        if split[i].strip() not in idxs[i]:
                            idxs[i][split[i].strip()] = len(idxs[i])
    return idxs;

def load_file(input_file,input_file_seperator,embeddings_index,idxs=None,max_len=0,considerFeatures=None):
    idxs = load_idxs(input_file, input_file_seperator,considerFeatures,max_len, idxs);

    
    max_char_len = 20

    file = open(input_file, "r")
    
    features = {}
    features["wordembeddings_input"]=[]
    features["char_input"]=[]
    features["output"]=[]
    features["words"]=[]
    
    features_current_sequence = {}
    features_current_sequence["wordembeddings_current_sequence"] = []
    features_current_sequence["char_current_sequence"] = []
    features_current_sequence["output_current_sequence"] = []
    features_current_sequence["words_current_sequence"] = []
    
    for i in considerFeatures:
        features_current_sequence[i]=[]
        features[i]=[]

    for line in file:
        line= line.strip()
        split = line.split(input_file_seperator)
        word = split[0]
        #sequence is finished when empty line is detected.
        if len(split) < 2:
            #skipping emtpy sequences.
            if(len(features_current_sequence["wordembeddings_current_sequence"]) < 1):
                for i in features_current_sequence:
                    features_current_sequence[i]=[]
                continue

            #adding padding vectores.
            while(len(features_current_sequence["wordembeddings_current_sequence"]) < max_len):
                features_current_sequence["wordembeddings_current_sequence"].append(0)
                features_current_sequence["output_current_sequence"].append(np.zeros(len(idxs["output_idx"])))
                features_current_sequence["char_current_sequence"].append([0]*max_char_len)
                for i in considerFeatures:
                    features_current_sequence[i].append(np.zeros(len(idxs[i])))

            #appending current sequence to hole sequence
            features["wordembeddings_input"].append(features_current_sequence["wordembeddings_current_sequence"][:max_len])
            features["char_input"].append(features_current_sequence["char_current_sequence"][:max_len])
            features["output"].append(features_current_sequence["output_current_sequence"][:max_len])
            features["words"].append(features_current_sequence["words_current_sequence"][:max_len])
            
            
            #resettin current sequence.
            for i in considerFeatures:
                features[i].append(features_current_sequence[i][:max_len])
            
            #resettin current sequence.
            for i in features_current_sequence:
                features_current_sequence[i]=[]
        else:
            char_input_line_char = []
            #create charinput
            for char in list(word):
                if(char in idxs["char_idx"]):
                    char_input_line_char.append(idxs["char_idx"][char])
                else:
                    char_input_line_char.append(0)
            while(len(char_input_line_char) < max_char_len):
                char_input_line_char.append(0)
            features_current_sequence["char_current_sequence"].append(char_input_line_char[:max_char_len])
            
            #create wordinput
            if(word.lower() in embeddings_index):
                features_current_sequence["wordembeddings_current_sequence"].append(embeddings_index[word.lower()])
            #if word not in wordembeddings input
            else:
                features_current_sequence["wordembeddings_current_sequence"].append(1)
            
            
            out = np.zeros(len(idxs["output_idx"]))
            if(split[1] in idxs["output_idx"]):
                out[idxs["output_idx"][split[1]]] = 1
            else:
                out[idxs["output_idx"]["O"]] = 1
            features_current_sequence["output_current_sequence"].append(out) #Oder 0. Aber wahrscheinlich eher 1
            features_current_sequence["words_current_sequence"].append(split[0])
            
            for i in considerFeatures:
                outTmp = np.zeros(len(idxs[i]))
                if(split[i] in idxs[i]):
                    outTmp[idxs[i][split[i]]] = 1
                features_current_sequence[i].append(outTmp) #Oder 0. Aber wahrscheinlich eher 1

    
    return features,idxs,max_len

def getKeyByValue(dict,value):
    return dict.keys()[dict.values().index(value)] # Prints george

def reconstructData(features,idxs):
    for sequence in range(0,len(features["words"])):
        out = ""
        for i in range(0,len(features["words"][sequence])):
            out = features["words"][sequence][i] 
            if(features["wordembeddings_input"][sequence][i]>1):
                out = out + "\t" + getKeyByValue(embeddings_index,features["wordembeddings_input"][sequence][i]) 
            else:
                out = out + "\t" + "NONE"
            out = out +"\t"+ getKeyByValue(idxs["output_idx"],np.argmax(features["output"][sequence][i]))
            for j in features:
                if j != "words" and j != "output" and j != "char_input" and j != "wordembeddings_input":
                    out = out + "\t" +getKeyByValue(idxs[j],np.argmax(features[j][sequence][i]))
            print out
        print 


def loadData(inputfile,embeddingFile, embeddingSkipHeader=False,considerFeatures=None,max_len=425):
    embeddings_matrix,embeddings_index = load_embeddings(embeddingFile)
    features,idxs,max_len = load_file(inputfile,"\t",embeddings_index,considerFeatures=considerFeatures,max_len=max_len)
    return features,idxs,embeddings_matrix,embeddings_index,max_len 

def test_conll(model, test_input, test_output, rev_classes,test_words = None):
    output = ""
    predict = model.predict(test_input)
    print("---------------")
    output_return = []
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            if(not test_output[i][j].any()):
                break
            max_pred_idx = np.argmax(predict[i][j])
            max_orig_idx = np.argmax(test_output[i][j])
            if test_words == None:
                output = output + "Token" + "\t" + str(rev_classes[max_orig_idx]) + "\t" + str(
                    rev_classes[max_pred_idx]) + "\n"
            else:
                output = output + test_words[i][j]+"\t" + str(rev_classes[max_orig_idx])+ "\t" + str(rev_classes[max_pred_idx]) + "\n"
            output_return.append(str(rev_classes[max_pred_idx]))
        output_return.append("")
        
    
    precision, recall, f1 = conlleval.global_evaluate(output)
    precision = precision/100.
    recall = recall/100.
    f1 = f1/100.
    print precision, recall, f1
    return precision, recall, f1,output_return

def batch_iter(data,data1, labels, batch_size, shuffle=True, preprocessor=None):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_data1 = data1[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_data1 = data1
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                [np.array(words_train), np.array(char_train)], np.array(y_train)
                X, y = [np.array(shuffled_data[start_index: end_index]),np.array(shuffled_data1[start_index: end_index])], np.array(shuffled_labels[start_index: end_index])
                if preprocessor:
                    yield preprocessor.transform(X, y)
                else:
                    yield X, y

    return num_batches_per_epoch, data_generator()

def createModel(consideredFeatures,embeddings_matrix,idxs,weights_path):
    char_emb_size = 25
    char_lstm_size = 25
    conditional = [True,False]
    conditional_1 = [True,False]
    lstm_units = [200, 50,100]
    
    params = loadJsonParams(weights_path.replace("h5", "json"))
    dropout = params["Dropout"][0]
    dropout_1 = params["Dropout_1"][0]
    conditional = conditional[params["conditional"][0]]
    conditional_1 = conditional_1[params["conditional_1"][0]]
    lstm_units = lstm_units[params["lstm_units"][0]]
    
    print dropout,dropout_1,conditional,conditional_1,lstm_units
    
    if embeddings_matrix is None:
        emb_layer = Embedding(input_dim=len(word_idx)+1,
                                    output_dim=300,
                                    mask_zero=True,
                                    trainable=False,
                                    name="word_emb")
    else:
        emb_layer = Embedding(input_dim=embeddings_matrix.shape[0],
                                    output_dim=embeddings_matrix.shape[1],
                                    mask_zero=True,
                                    weights=[embeddings_matrix],
                                    name="word_emb",
                                    trainable=False)

    word_ids = Input(batch_shape=(None, None), dtype='int32')
    word_embeddings = emb_layer(word_ids)


    #     #Char embeddings
    char_ids = Input(batch_shape=(None, None, None), dtype='int32')

    char_embeddings = Embedding(input_dim=len(idxs["char_idx"])+1,
                                output_dim=char_emb_size,
                                mask_zero=True,
                                input_length=20,
                                name = "char_emb"
                                )(char_ids)    
    s = K.shape(char_embeddings)

    if conditional == True:
    #     #####Attentionlayer
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, 20, char_emb_size)),name="lambda")(char_embeddings)
        fwd_state = LSTM(char_lstm_size, return_sequences=True, name="char_for", )(char_embeddings)
        bwd_state = LSTM(char_lstm_size, return_sequences=True, go_backwards=True, name="char_back")(char_embeddings)
        char_embeddings = Concatenate(axis=-1, name="char_concat")([fwd_state,bwd_state])
        char_embeddings = attention_3d_block(char_embeddings,20,"char_att")
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], char_lstm_size*2]), name="lambda2")([char_embeddings])
    else:
        #Original
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], char_emb_size)),name="lambda")(char_embeddings)
        fwd_state = LSTM(char_lstm_size, return_state=True, name="char_for", )(char_embeddings)[-2]
        bwd_state = LSTM(char_lstm_size, return_state=True, go_backwards=True, name="char_back")(char_embeddings)[-2]
        #Charembeddings concatinate forward and backword
        char_embeddings = Concatenate(axis=-1, name="char_concat")([fwd_state, bwd_state])
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * char_lstm_size]), name="lambda2")(char_embeddings)
    
    
    
    
    embeddings = [word_embeddings, char_embeddings]
    additional_features = []
    additional_features_length = 0
    for j in consideredFeatures:
        if j != "words" and j != "output" and j != "char_input" and j != "wordembeddings_input":
            additional_features_length = additional_features_length + len(idxs[j])
            additional_features .append(Input(batch_shape=(None, None,len(idxs[j])), dtype='float32'))
    
    if conditional_1 == True:
        additional_features_concat = Concatenate(axis=-1, name="concat_additional_feats")(additional_features)
        print additional_features_concat.shape
        attention_probs = Dense(additional_features_length, activation='softmax', name='attention_vec_')(additional_features_concat)
        attention_mul = Multiply()([additional_features_concat, attention_probs])
        print word_embeddings.shape, char_embeddings.shape
        embeddings.append(attention_mul)
        merge_input = Concatenate(axis=-1, name="concat_all")(embeddings)
    else:
        merge_input = Concatenate(axis=-1, name="concat_all")(embeddings+additional_features)
    
    merge_input = Dropout(dropout)(merge_input)

    merge_lstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True, name="lstm1"), name="bilstm")(merge_input)
    
    merge_lstm1 = Dropout(dropout_1)(merge_lstm1)
#         merge_lstm1 = Dense(100, activation="tanh")(merge_lstm1)
    merge_lstm1 = Dense(len(idxs["output_idx"]))(merge_lstm1)
    
    
    crf = CRF(len(idxs["output_idx"]))
    pred = crf(merge_lstm1)
    lossFct = crf.loss_function
    model = Model(inputs=[word_ids, char_ids]+additional_features, outputs=[pred])
    model.compile(loss=lossFct,optimizer=Adam(lr=0.001), metrics=[crf.accuracy])
    
    return model
            
def train(model, features_train,embeddings_matrix,idxs,weights_path):
    inputs = [np.array(features_train["wordembeddings_input"]), np.array(features_train["char_input"])]
    for j in features_train:
        if j != "words" and j != "output" and j != "char_input" and j != "wordembeddings_input":
            inputs.append(np.array(features_train[j]))
            
    loss_function = "viterbi_acc"
    if not os.path.exists(weights_path):
        os.mkdir( weights_path, 0755 )
    checkpointCallback = keras.callbacks.ModelCheckpoint(
        weights_path+"/weights.{epoch:02d}-{"+loss_function+":.2f}.hdf5", 
        monitor=loss_function, 
        verbose=0, 
        save_weights_only=True, 
        mode='auto', 
        period=1)
    
    earlystopCallback = keras.callbacks.EarlyStopping(monitor=loss_function,
                              min_delta=0,
                              patience=5,
                              verbose=2, mode='auto')

    model.fit(inputs, np.array(features_train["output"]), batch_size=128, epochs=100, verbose=1 
                            ,callbacks=[earlystopCallback,checkpointCallback])
    model.save_weights(weights_path+"/last_model.hdf5")


def test(model,consideredFeatures,features_test,idxs,weights_path):
    print weights_path;
    model.load_weights(weights_path)
    rev_classes = ["-"]*len(idxs["output_idx"])
    for key in idxs["output_idx"]:
        rev_classes[idxs["output_idx"][key]] = key
    inputs_test = [np.array(features_test["wordembeddings_input"]), np.array(features_test["char_input"])]
    for j in consideredFeatures:
        if j != "words" and j != "output" and j != "char_input" and j != "wordembeddings_input":
            inputs_test.append(np.array(features_test[j]))
    return test_conll(model, inputs_test, np.array(features_test["output"]), rev_classes, features_test["words"])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-test", "--test", action='store_true',
                        help="test modus")
    parser.add_argument("-train", "--train", action='store_true',
                        help="train modus")
    parser.add_argument("-m", "--model", help="Path to modelfile")

    parser.add_argument("-tf", "--testfile", help="Path to testfile")
    
    args = parser.parse_args()
    
    consideredFeatures=[5,6,7,8]
    
    max_len = 425
    
    if args.test and args.train:
        raise Exception("specifiy either train or test")
    elif args.test:
        embeddings_matrix,embeddings_index = load_embeddings("python/embeddings/chemdner.txt")
        
        with open(args.model.replace("h5", "idxs"), 'rb') as handle:
            idxs = pickle.load(handle)
        #         with open('file', 'wb') as handle:
        #             pickle.dump(idxs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #         print pickle.dumps(idxs)
        model = createModel(consideredFeatures,embeddings_matrix,idxs,args.model)

        features_test,idxs,max_len = load_file(args.testfile,"\t",embeddings_index,considerFeatures=consideredFeatures,max_len=max_len,idxs=idxs)
        precision, recall, f1,output_return = test(model,consideredFeatures,features_test,idxs,args.model)
              
        advancedOutput = conllToAdvanced(args.testfile, output_return);
         
        biocreativeOutputString = StringIO()
        for i in advancedOutput:
            biocreativeOutputString.write(i + "\n")
             
        biocreativeOutputString = biocreativeOutputString.getvalue()
        with open("output.txt", "w") as text_file:
            text_file.write(biocreativeOutputString)
                 
        biocreativeOutput = advancedToBioCreative(advancedOutput)
                 
        simpleOutput = biocreativeToSimpleBioCreative(biocreativeOutput, True)
        with open("evaluation.txt", "w") as text_file:
            text_file.write(simpleOutput)
    elif args.train:
        features_train,idxs,embeddings_matrix,embeddings_index,max_len = loadData(trainFile,"/home/staff_homes/ahemati/projects/biocreative/data/embeddings/chemdner.txt",False,considerFeatures=consideredFeatures,max_len=max_len)
        train(model,features_train,embeddings_matrix,idxs,weights_path)
    else:
        raise Exception("specifiy either train or test")
