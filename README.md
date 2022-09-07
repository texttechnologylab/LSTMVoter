[![version](https://img.shields.io/github/license/texttechnologylab/LSTMVoter)]()
[![](https://jitpack.io/v/texttechnologylab/LSTMVoter.svg)](https://jitpack.io/#texttechnologylab/LSTMVoter)
[![Paper](http://img.shields.io/badge/paper-Journal_of_Cheminformatics-B31B1B.svg)](https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-018-0327-2.pdf)


# Abstract
Chemical and biomedical named entity recognition (NER) is an essential preprocessing task in natural language processing. The identification and extraction of named entities from scientific articles is also attracting increasing interest in many scientific disciplines. Locating chemical named entities in the literature is an essential step in chemical text mining pipelines for identifying chemical mentions, their properties, and relations as discussed in the literature. In this work, we describe an approach to the BioCreative V.5 challenge regarding the recognition and classification of chemical named entities. For this purpose, we transform the task of NER into a sequence labeling problem. We present a series of sequence labeling systems that we used, adapted and optimized in our experiments for solving this task. To this end, we experiment with hyperparameter optimization. Finally, we present LSTMVoter, a two-stage application of recurrent neural networks that integrates the optimized sequence labelers from our study into a single ensemble classifier.

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
python python/LSTMVoter_infer.py -test -m python/models/LSTMVoter/hyperas/model1.h5 -tf python/data/CHEMDNER/evaluation_taged_by_multitagger.tiny.conll
```

# Cite
If you want to use the project please quote this as follows:

 W. Hemati and A. Mehler, “LSTMVoter: chemical named entity recognition using a conglomerate of sequence labeling tools,” Journal of Cheminformatics, vol. 11, iss. 1, p. 7, 2019. ![[Link]](https://doi.org/10.1186/s13321-018-0327-2) ![[PDF]](https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-018-0327-2.pdf)
 
## BibTeX
```
@article{Hemati:Mehler:2019a,
  author = "Hemati, Wahed and Mehler, Alexander",
  day = "10",
  doi = "10.1186/s13321-018-0327-2",
  issn = "1758-2946",
  journal = "Journal of Cheminformatics",
  month = "Jan",
  number = "1",
  pages = "7",
  title = "{{LSTMVoter}: chemical named entity recognition using a conglomerate of sequence labeling tools}",
  url = "https://doi.org/10.1186/s13321-018-0327-2",
  volume = "11",
  year = "2019"
}
