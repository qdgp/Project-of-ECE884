import os
import shutil
import pandas as pd
import numpy as np

path='./levels'
if not os.path.exists(path):
    os.mkdir(path)
labels = pd.read_csv('./trainLabels_cropped.csv')

for f in labels.level.unique():
    os.mkdir(path+'/{}'.format(f))
num=np.zeros(labels.level.unique().size)
for f in labels.level.unique():
    df = labels.loc[labels['level'] == f]
    l = df['image'].tolist()
    for im in l:
        if num[f]<700:
            num[f]=num[f]+1
            shutil.copy('./resized_train_cropped/resized_train_cropped/{}.jpeg'.format(im), path+'/{}'.format(f))
