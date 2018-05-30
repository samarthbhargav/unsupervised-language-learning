# README - ULL: Practical 1
By, Samarth Bhargav and Tharangni Harsha Sivaji

Requires the setup of the requirements present in requirements.txt file present in the root of the repository, which can be done by executing the following command (assumes Python3.5x and pip is installed):
```
pip install -r requirements.txt
pip install gensim pandas # additional requirements for Practical-3

# Install dgm4nlp from https://github.com/uva-slpl/dgm4nlp/
# Install SentEval from https://github.com/facebookresearch/SentEval
```

## 1. Training / importing models

We've already run the evaluation, and committed the `.pkl` files required for producing plots & tables - if you want to just use these files (and not run the evaluation and train Skipgram), skip to Section 2.2

### 1.1 Skipgram model
The Skipgram model needs to be trained on the English-Europarl data:

- Change the variable `EUROPARL_DATA` to point to Europarl `training.en`
- If required, change the parameters in the `get_skipgram` function. It already has reasonable defaults corresponding to the values mentioned in the report.
```
SKIPGRAM=<location of the model>
python skipgram.py $SKIPGRAM
```

### 1.2 Embed Align model

Download the pre-trained model to path. The next section assumes that the path is saved in `EMBED_ALIGN`

# 2. Evaluation

## 2.1 Evaluating the models

To evaluate both models on the SentEval tasks:
- If required, change the following variables in the `evaluate_models.py` script:
- `PATH_TO_SENTEVAL`. Default is `./SentEval`
- Edit the `run_all.sh` script variables to point to the right location
- Run `run_all.sh`
- The `.pkl` files containing the results are saved in `./results_min`, `./results_max`, `./results_avg` and `./results_concat`

## 2.2 Visualization and viewing the performance
Fire up `jupyter` and open `viz.ipynb` and run the code. It should create (a) create the required plots (b) print out the performance of each model and composition method
