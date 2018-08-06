## Installation
```
extract python/embeddings/chemdner.txt.zip
```


```bash
$ pip install -r python/requirements.txt
```

## Usage

```bash
python python/tagger_attention_keras_test.py -test -m python/models/LSTMVoter/hyperas/model1.h5 -tf python/data/CHEMDNER/evaluation_taged_by_multitagger.tiny.conll
```
