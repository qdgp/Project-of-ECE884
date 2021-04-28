import os
import shutil
import pandas as pd
import numpy as np

if not os.path.exists('./levels'):
    os.mkdir('./levels')
labels = pd.read_csv('./archive/trainLabels_cropped.csv')

for f in labels.level.unique():
    os.mkdir('./levels/{}'.format(f))
num=np.zeros(labels.level.unique().size)
for f in labels.level.unique():
    df = labels.loc[labels['level'] == f]
    l = df['image'].tolist()
    for im in l:
        if num[f]<700:
            num[f]=num[f]+1
            shutil.copy('./archive/resized_train_cropped/resized_train_cropped/{}.jpeg'.format(im), './levels2/{}'.format(f))
