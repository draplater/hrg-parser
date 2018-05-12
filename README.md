# HRG Parser

## Recommended Environment 
- Linux
- Java 8
- Python 3.6
- GCC 7.0

## Data Preparation
You need get DeepBank1.1 data first.

### Split the punctuation from the lexicon
Compile the java project "pseud" and run:
```java jigsaw.treebank.RedwoodsTreeReader <"profile" dir of deepbank1.1> ```

The outputs are in ```out``` directory. You should manually split them into train, dev and test set into directory ```java_out_train```, ```java_out_dev```, ```java_out_test``` respectively, and put them into a common directory ```main_dir_base```.

### install python dependency
Run ```pip3 install -r requirements.txt```

### Extract grammar
Modify ```deepbank_export_path``` and ```main_dir_base``` in ```extract_sync_grammar.py``` and run it. The output trees, derivation and grammar will be in ```deepbank-preprocessed``` directory.

### Train a POSTagger
Modify paths in ```scripts/train_leaftag.py``` and run it.

### Train a Phrase Structure Parser
Modify paths in ```scripts/train_span_lite.py``` and run it.

### Predict with Trained Phrase Structure Parser
Run ```main.py span predict``` and follow the instruction.

### Train a HRG Parser
Modify paths in ```scripts/train_udef_lite.py``` and run it.

### Predict with Trained HRG Parser
Run ```main.py hrg predict``` and follow the instruction.
