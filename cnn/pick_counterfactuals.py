import sys
import os
import glob
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np

# pull most recent probabilities from cnn
cnn_dir = Path('cnn')
csv_files = glob.glob(str(cnn_dir / 'predicted_activation-*.csv'))
if not csv_files:
    raise FileNotFoundError("No CSV files found in the 'cnn' directory.")
most_recent_csv = max(csv_files, key=os.path.getmtime)
probabilities = pd.read_csv(most_recent_csv)

target_prob = probabilities.loc[probabilities['real_missing'] == True,'prob_1'].median()
print(f'target probability is {target_prob}')

counterfactuals = probabilities.loc[(probabilities['prob_1'] >= 0.99) & (probabilities['real_missing'] == False), ['s_id', 'city']]
print(f'found {len(counterfactuals)} counterfactuals with prob >= {0.99}')

counterfactual_dict = {row['city']: set() for _, row in counterfactuals.iterrows()}
for _, row in counterfactuals.iterrows():
    counterfactual_dict[row['city']].add(row['s_id'])

print(probabilities['prob_1'].describe(), probabilities['prob_0'].describe())
