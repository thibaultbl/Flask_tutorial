#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:59:32 2017

@author: thibault
"""
import pandas as pd
import requests
import json

PATH_DATA = "/home/thibault/Documents/Flask/Flask_test/Data/"
PATH_MODEL= "/home/thibault/Documents/Flask/Flask_test/Model/"
FILENAME = 'model_v1.pk'

 
"""Setting the headers to send and accept json responses
"""     
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

"""Reading test batch
"""
df = pd.read_csv(PATH_DATA + 'test.csv', encoding="utf-8-sig")
df = df.head()

"""Converting Pandas Dataframe to json
"""
data = df.to_json(orient='records')

resp = requests.post("http://127.0.0.1:5000/predict", \
                    data = json.dumps(data),\
                    headers= header)

resp.status_code

resp.json()