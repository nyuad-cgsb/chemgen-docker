#!/usr/bin/env python3

import os
import glob
import requests

uri = 'http://pyrite.abudhabi.nyu.edu:3005/tf_counts/1.0/api/count_worms'


def push_counts_to_queue(dirs):
    print('Pushing {}'.format(os.path.dirname(dirs[0])))
    for d in dirs:
        data = {}
        b = os.path.basename(d)
        data['image_path'] = d
        data['counts'] = '{}/{}-tf_counts.csv'.format(d, b)
        requests.get(uri, json=data, timeout=2000)
    print('Finished {}'.format(os.path.dirname(dirs[0])))


dirs = glob.glob('/mnt/image/20*')
for d in dirs:
    if os.path.isdir(d):
        ds = glob.glob('{}/*'.format(d))
        push_counts_to_queue(ds)

# dirs2018 = glob.glob('/mnt/image/2018*/*')
# push_counts_to_queue(dirs2018)
# dirs2017 = glob.glob('/mnt/image/2017*/*')
# push_counts_to_queue(dirs2017)
# dirs2016 = glob.glob('/mnt/image/2016*/*')
# push_counts_to_queue(dirs2016)
# dirs2015 = glob.glob('/mnt/image/2015*/*')
# push_counts_to_queue(dirs2015)
# dirs2014 = glob.glob('/mnt/image/2014*/*')
# push_counts_to_queue(dirs2014)

