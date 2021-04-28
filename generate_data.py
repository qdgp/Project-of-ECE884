import os
import splitfolders 
import shutil
import pandas as pd

if not os.path.exists('./levels'):
    os.mkdir('./levels')
labels = pd.read_csv('./archive/trainLabels_cropped.csv')

for f in labels.level.unique():
    if not os.path.exists('./levels/{}'.format(f)):
        os.mkdir('./levels/{}'.format(f))

for f in labels.level.unique():
    df = labels.loc[labels['level'] == f]
    l = df['image'].tolist()
    for im in l:
        shutil.copy('./archive/resized_train_cropped/resized_train_cropped/{}.jpeg'.format(im), './levels/{}'.format(f))

        
#for f in labels.level.unique():
#    os.mkdir('./levels2/{}'.format(f))
#num=np.zeros(labels.level.unique().size)
#for f in labels.level.unique():
#    df = labels.loc[labels['level'] == f]
#    l = df['image'].tolist()
#    for im in l:
#        if num[f]<700:
#            num[f]=num[f]+1
#            shutil.copy('./archive/resized_train_cropped/resized_train_cropped/{}.jpeg'.format(im), './levels2/{}'.format(f))

        
input_dir = os.path.join('./levels/')
output_dir = os.path.join('./levels_splitted/')
splitfolders.ratio(input_dir, output=output_dir, seed=1337, ratio=(.8, .2), group_prefix=None)

