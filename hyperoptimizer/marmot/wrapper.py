from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from hyperopt.mongoexp import MongoTrials
import marmot
import json
import os

trainFile = os.path.abspath("../../data/CHEMDNER/first_step_train/training.small.conll.txt")
testFile = os.path.abspath("../../data/CHEMDNER/first_step_train/training.small.conll.txt")

space4knn = {
    "trainFile":trainFile,
    "testFile":testFile,
    "num-iterations":hp.choice('num-iterations', [10,50,100]),
    "optimize-num-iterations":hp.choice('optimize-num-iterations', ["true","false"]),
    "beam-size":hp.choice('beam-size', [1,2,3,4,5]),
    "initial-vector-size":hp.choice('initial-vector-size', [10000000,1000000,50000000]),
    "prune":hp.choice('prune', ["true","false"]),
    "order":hp.choice('order', [1,2,3,4,5]),
}
trials = Trials()
best = fmin(marmot.f, space4knn, algo=tpe.suggest, max_evals=10, trials=trials)
print 'best:'
print best
