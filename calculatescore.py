import pandas as pd
import numpy as np
from math import log
from sklearn.metrics import f1_score, recall_score, precision_score

def preprocess_row(row):
    row = [max(min(p, 1 - 10e-15), 10e-15) for p in row]
    row = [x/sum(row) for x in row]
    return row

def calculate_score(answers_path, submission_path):
    right_answers = pd.read_csv(answers_path)
    sam_submission = pd.read_csv(submission_path)

    y = right_answers.values[:,1:].astype(int)
    submission = sam_submission.values[:,1:].astype(float)

    submission = np.apply_along_axis(func1d=preprocess_row, axis=1, arr=submission)

    submission = np.log(submission)
    temp = np.multiply(submission, y)

    return np.sum(temp)/-submission.shape[0]


def evaluate(gold_path, predictions_path):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    # different ways of averaging
    predictions = pd.read_csv(predictions_path)
    predictions = predictions.values[:,1:].astype(int)
    
    gold = pd.read_csv(gold_path)
    gold = gold.values[:,1:].astype(int)

    f1 = f1_score(gold, predictions, average=None)
    precision = precision_score(gold, predictions, average=None)
    recall = recall_score(gold, predictions, average=None)
    
    return precision, recall, f1

def eval_gender(gold_path, pred_path, full_data_path):
    
    # Full data #
    df = pd.read_csv(full_data_path, sep = '\t', index_col = 0)
    
    # Define pronouns #
    # Male pronouns
    male_pn = ['He', 'His', 'he', 'him', 'his']

    # Female pronouns
    female_pn = ['Her', 'She', 'her', 'she']
    
    # Find list of indices
    m_idx = []
    fm_idx = []
    for i in range(df.shape[0]):
        pn = df['Pronoun'][i]
        if pn in male_pn:
            m_idx.append(list(df.index)[i])
        else:
            fm_idx.append(list(df.index)[i])
            
    # Import data #
    predictions = pd.read_csv(pred_path, index_col = 0)
    
    # Import gold #
    gold = pd.read_csv(gold_path, index_col = 0)    
    
    # predictions = predictions.values[:, :].astype(int)
    # gold = gold.values[:, :].astype(int)

    # Male pred #
    male_pred = predictions.loc[m_idx, :]
    male_pred = male_pred.values[:, :].astype(int)
    male_gold = gold.loc[m_idx, :]
    male_gold = male_gold.values[:, :].astype(int)

    # Female pred
    female_pred = predictions.loc[fm_idx, :]
    female_pred = female_pred.values[:, :].astype(int)
    female_gold = gold.loc[fm_idx, :]
    female_gold = female_gold.values[:, :].astype(int)

    # Male scores
    f1_m = f1_score(male_gold, male_pred, average = None)
    precision_m = precision_score(male_gold, male_pred, average = None)
    recall_m = recall_score(male_gold, male_pred, average = None)

    # Female scores
    f1_fm = f1_score(female_gold, female_pred, average = None)
    precision_fm = precision_score(female_gold, female_pred, average = None)
    recall_fm = recall_score(female_gold, female_pred, average = None)

    # Print values #
    print('Male precision: A = %s, B = %s, Neither = %s' % tuple(precision_m.round(4)))
    print('Male recall: A = %s, B = %s, Neither = %s' % tuple(recall_m))
    print('Male F1: A = %s, B = %s, Neither = %s \n' % tuple(f1_m.round(4)))

    print('Female precision: A = %s, B = %s, Neither = %s' % tuple(precision_fm))
    print('Female recall: A = %s, B = %s, Neither = %s' % tuple(recall_fm))
    print('Female F1: A = %s, B = %s, Neither = %s' % tuple(f1_fm.round(4)))

    # Male dict #
    male = {}
    male['precision'] = precision_m
    male['recall'] = recall_m
    male['f1'] = f1_m

    # Female dict #
    female = {}
    female['precision'] = precision_fm
    female['recall'] = recall_fm
    female['f1'] = f1_fm

    # Return
    return male, female
