import glob, os
import numpy as np
import matplotlib.pyplot as plt
from image_prep import prep_image
from keras.models import load_model
import pandas as pd

X = []
idx = []
counter = 0
for f in glob.glob('test_stg1/*.jpg'):
    idx.append(f[10:])
    X.append(prep_image(plt.imread(f)))
    print f, 'complete' 
print len(X), 'stage 1 samples'

np.save('stg1_prcssd.npy', X)

counter = 0
for f in glob.glob('test_stg2/*.jpg'):
    idx.append(f[10:])
    X.append(prep_image(plt.imread(f)))
    print f, 'complete'
X = np.array(X)
print X.shape[0], 'stage1 and stage2 samples'

np.save('both_prcssd.npy', X)

model = load_model('models/ns_mod4.h5')

proba = model.predict_proba(X)

probs = pd.DataFrame(proba)

probs['image'] = idx

probs.set_index('image', drop=True, inplace=True)

cols = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

probs.columns = cols

probs.to_csv('probs.csv')
