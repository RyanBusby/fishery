import os
import glob
from boto.s3.connection import S3Connection
import boto3
import numpy as np
import matplotlib.pyplot as plt
from imprep import *

np.random.seed(11)

conn = S3Connection()
s3 = boto3.resource('s3')

pics = []
y = []

for n, species in enumerate(['alb', 'bet', 'dol', 'lag', 'nof', 'other', 'shark', 'yft']):
    name = 'rb-fishery-'+species
    bucket = conn.get_bucket(name)
    key_list = bucket.get_all_keys(max_keys=5)
    keys = []
    for key in key_list:
        keys.append(key.key)
    for key in keys:
        try:
            s3.meta.client.download_file(name, key, 'temp/'+str(key))
            im = plt.imread('temp/'+str(key))
            x = prep_image(im)
            pics.append(x)
            y.append(n)
            print key, 'complete'
            os.system('rm temp/'+str(key))
        except Exception as ex:
            print ex
            os.system('rm temp/'+str(key))
            print key, 'not downloaded'
            continue

#X = np.array(pics)
np.save('temp/test_prep_X.npy', pics)
np.save('temp/test_prep_y.npy', y)
#os.system('''aws s3 cp 'temp/' s3://rb-fishery-python-objects/ --recursive''')
#os.system('rm temp/*.npy')
