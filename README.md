## Dataset Converter

### Prerequisites
In order to run IOB Converter you need
* Java 8
* Maven

### Installing
Clone and star this repository
```
git clone https://github.com/texttechnologylab/LSTMVoter
```
Navigate to the directory and build project
```
cd ../some/dir/LSTMVoter
mvn install -DskipTests
```

### Running

#### CLI API
After ```mvn install``` the CLI script will be generated in ```target/converter.jar```.

Example client call:
```
java -jar target/converter.jar [abstractsFile] [annotationsFile] [outputFile]
```

## Installation LSTMVoter
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
