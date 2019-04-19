import pandas as pd
import numpy as np
from math import log

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