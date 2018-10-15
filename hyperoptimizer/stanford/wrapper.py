from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from hyperopt.mongoexp import MongoTrials
import stanford
import json
import os

trainFile = os.path.abspath("../../data/CHEMDNER/conll/first_step_train/training.small.conll.txt")
devFile = os.path.abspath("../../data/CHEMDNER/conll/first_step_train/train_development.conll.txt")

space4knn = {
    "map" : "word=0,answer=1",
    "entitySubclassification":"IOB1",
    # these are the features we'd like to train with
    # some are discussed below, the rest can be
    # understood by looking at NERFeatureFactory
    "useClassFeature":"true",
    "useWord":"true",
    "useNGrams":"true",
    "noMidNGrams":hp.choice('noMidNGrams', ["true","false"]),
    "normalizeTerms":hp.choice('normalizeTerms', ["true","false"]),
    "usePosition":hp.choice('usePosition', ["true","false"]),
      
    "useNeighborNGrams":hp.choice('useNeighborNGrams', ["true","false"]),
    "useMoreNeighborNGrams":hp.choice('useMoreNeighborNGrams', ["true","false"]),
      
    "usePrev":hp.choice('usePrev', ["true","false"]),
    "useNext":hp.choice('useNext', ["true","false"]),
    "useTags":hp.choice('useTags', ["true","false"]),
    "useWordPairs":hp.choice('useWordPairs', ["true","false"]),
  
    "useDisjunctive":hp.choice('useDisjunctive', ["true","false"]),
    "useSequences":hp.choice('useSequences', ["true","false"]),
    "usePrevSequences":hp.choice('usePrevSequences', ["true","false"]),
    "useNextSequences":hp.choice('useNextSequences', ["true","false"]),
    "useLongSequences":hp.choice('useLongSequences', ["true","false"]),
    "useTaggySequences":hp.choice('useTaggySequences', ["true","false"]),
  
    "useSymWordPairs":hp.choice('useSymWordPairs', ["true","false"]),
    "useSymTags":hp.choice('useSymTags', ["true","false"]),
  
  
    "useTypeSeqs":hp.choice('useTypeSeqs', ["true","false"]),
    "useTypeSeqs2":hp.choice('useTypeSeqs2', ["true","false"]),
    "useTypeySequences":hp.choice('useTypeySequences', ["true","false"]),
    "wordShape":"chris2useLC",
    "maxLeft": hp.choice('maxLeft', range(1,3)),
    "maxRight": hp.choice('maxRight', range(1,3)),
    "maxNGramLeng": hp.choice('maxNGramLeng', range(2,6)),
    "types" :".type1",
#     "gazetteOptions":hp.choice('gazetteOptions',
#                         [
#                             {"useGazettes":"false"},
#                             {"useGazettes":"true","gazette":hp.choice('gazette', ["/home/staff_homes/ahemati/projects/biocreative/data/gazette/cemp.gazette","/home/staff_homes/ahemati/projects/biocreative/data/gazette/cemp.type1.gazette"]),"sloppyGazette":hp.choice('sloppyGazette', ["true","false"])}
#                         ]
#                         ),
#     "useGazFeatures":hp.choice('useGazFeatures', ["true","false"]),

    "trainFile":trainFile,
    "testFile":devFile,
     
    "useWordTag":hp.choice('useWordTag', ["true","false"]),
    "useWideDisjunctive":hp.choice('useWideDisjunctive', ["true","false"]),
    "useLemmas":"true",
    "usePrevNextLemmas":"true"
}
trials = Trials()
best = fmin(stanford.f, space4knn, algo=tpe.suggest, max_evals=10, trials=trials)
print 'best:'
print best
