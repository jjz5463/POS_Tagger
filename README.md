# Homework 2 of Jiacheng Zhu & Jingyi Guo

## Run File

use `python pos_tagger.py -h` to ask for help menu to run postagger

there are four argument
  
  - `-g` grams (1,2,3,4)
  - `-a` algorithm ("greedy", "beam", or "viterbi", gram = 4 doesn't support viterbi)
  - `-s` smooth ("k" or "linear". optional, default to add-k. only support for gram = 3)
  - `-t` test file path (optional, if input, will output prediction on test to path)

use `python evaluate.py -h` to ask for help menu to create confusion matrix for the leaderboard result

examples:

1. evaluate 2-grams viterbi on the dev set
```
python pos_tagger.py -g 2 -a viterbi
```

2. evaluate 3-grams viterbi on the dev set with add-k smooth
```
python pos_tagger.py -g 3 -a viterbi -s k
```

3. evaluate 3-grams viterbi on the dev set with add-k smooth and output test predictions
```
python pos_tagger.py -g 3 -a viterbi -s k -t ~/Desktop/test_y.csv
```

4. evaluate the dev_pred and dev_x to gave a confusion matrix (the one upload to leaderboard, 2-gram viterbi)
```
python evaluate.py -p data/dev_pred.csv -d data/dev_x.csv -c
```

## Files
There are 3 python scripts.
* `pos_tagger.py`: The complete postagger with 1,2,3,4 grams, viterbi, greedy, and beam search. 
*  `constants.py`: This file contains any constants needed for the assignment. You are not required to use it, but it might be helpful if you want to run multiple experiments with different settings. Alternatively, you could pass command line arguments using `argparse` (see [documentation](https://docs.python.org/3/library/argparse.html)). Note, if you think you should add more constants/hyperparameters feel free to do so. 
* `utils.py`:  This file contains utility functions that are used in `pos_tagger.py`. You can add more helper functions here for better organization of your code. 

## Folder Structure
please arrange file in following structure

- **hwk2_code_JGJZ (root directory)**
  - **Data (Directory)**
    - dev_pred.csv
    - dev_x.csv
    - dev_y.csv
    - test_x.csv
    - train_x.csv
    - train_y.csv
    - words_and_tags_comb.csv *(tokenized word for training for the unknown word classifier)*
  - **model (Directory)**
    - simplednn_model_4.pth *(for unknown word classification, trained on training data)*
  - **image (Directory)** *(empty folder to store confusion matrix)*
  - **pos_tagger.py** *(main file)*
  - **evaluate.py**
  - **utils.py**
  - **constants.py**
  - **requirements.txt**
  - **README.md**