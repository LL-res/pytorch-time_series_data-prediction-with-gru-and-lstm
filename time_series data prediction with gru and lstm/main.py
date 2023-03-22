import json
import sys
from datetime import datetime
from typing import List
STATUS_TRAIN =  1
STATUS_PREDICT = 2
class TimestampedMetrics:
    def __init__(self, time, metric, metric_type):
        self.time = time
        self.metric = metric
        self.metric_type = metric_type

class GRUParameters:
    def __init__(self, d):
       self.__dict__ = d

def predict(look_ahead,metrcis,type):
    print(look_ahead,metrcis,type)

def train(metrics,type):
    print(metrics,type)
# 00 predict train
def check_status(gru_params):
    status = 0
    if len(gru_params.train_history):
        status |= STATUS_TRAIN
    if len(gru_params.predict_history):
        status |= STATUS_PREDICT
    return status
str = '{"look_ahead": 300.0, "train_history": [{"time": "2022-03-22T10:00:00Z", "metric": 1.23, "type": "cpu"}], "predict_history": [{"time": "2022-03-22T11:00:00Z", "metric": 2.34, "type": "memory"}]}'
#sys.stdin.read()
gru_params = json.loads(str, object_hook=GRUParameters)

def Run(gru_params):
    if check_status(gru_params) & STATUS_PREDICT:
        metrics = []
        for index,val in enumerate(gru_params.predict_history):
            metrics.append(val.metric)
        predict(gru_params.look_ahead,metrics,gru_params.predict_history[0].type)
    if check_status(gru_params) & STATUS_TRAIN:
        metrics = []
        for index,val in enumerate(gru_params.train_history):
            metrics.append(val.metric)
        train(metrics,gru_params.train_history[0].type)

Run(gru_params)
