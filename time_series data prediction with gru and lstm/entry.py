import json
import sys
from datetime import datetime
from typing import List
from train import train
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

def predict_fake(look_ahead,metrcis,type):
    print(look_ahead,metrcis,type)

def train_fake(metrics,type):
    print(metrics,type)
# 00 predict train
def check_status(gru_params):
    status = 0
    if hasattr(gru_params,'train_history') and gru_params.train_history is not None and len(gru_params.train_history):
        status |= STATUS_TRAIN
    if hasattr(gru_params,'predict_history') and gru_params.predict_history is not None and len(gru_params.predict_history):
        status |= STATUS_PREDICT
    return status

str = '''
    {
        "look_ahead" : 300,
        "train_history": [
            {"time": "2022-03-22T10:00:00Z", "metric": 0.5, "type": "cpu"},
            {"time": "2022-03-22T10:00:15Z", "metric": 0.9, "type": "cpu"},
            {"time": "2022-03-22T10:00:30Z", "metric": 1.2, "type": "cpu"},
            {"time": "2022-03-22T10:00:45Z", "metric": 1.5, "type": "cpu"},
            {"time": "2022-03-22T10:01:00Z", "metric": 1.7, "type": "cpu"},
            {"time": "2022-03-22T10:01:15Z", "metric": 1.9, "type": "cpu"},
            {"time": "2022-03-22T10:01:30Z", "metric": 2.0, "type": "cpu"},
            {"time": "2022-03-22T10:01:45Z", "metric": 2.0, "type": "cpu"},
            {"time": "2022-03-22T10:02:00Z", "metric": 1.9, "type": "cpu"},
            {"time": "2022-03-22T10:02:15Z", "metric": 1.7, "type": "cpu"},
            {"time": "2022-03-22T10:02:30Z", "metric": 1.5, "type": "cpu"},
            {"time": "2022-03-22T10:02:45Z", "metric": 1.2, "type": "cpu"},
            {"time": "2022-03-22T10:03:00Z", "metric": 0.9, "type": "cpu"},
            {"time": "2022-03-22T10:03:15Z", "metric": 0.5, "type": "cpu"},
            {"time": "2022-03-22T10:03:30Z", "metric": 0.2, "type": "cpu"},
            {"time": "2022-03-22T10:03:45Z", "metric": 0.0, "type": "cpu"}
        ]
    }
'''
#sys.stdin.read()
gru_params = json.loads(str, object_hook=GRUParameters)

def Run(gru_params):
    if check_status(gru_params) & STATUS_PREDICT:
        metrics = []
        for index,val in enumerate(gru_params.predict_history):
            metrics.append(val.metric)
        predict_fake(gru_params.look_ahead,metrics,gru_params.predict_history[0].type)
    if check_status(gru_params) & STATUS_TRAIN:
        metrics = []
        for index,val in enumerate(gru_params.train_history):
            metrics.append(val.metric)
        train(metrics,gru_params.train_history[0].type)

Run(gru_params)
