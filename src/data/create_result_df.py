import pandas as pd
import numpy as np
import os

class CreateResult:
    def __init__(self, acc, model):
        self.acc = acc,
        self.model = model

    def create(self):
        if os.path.exists("../predictions/cv_results.csv"):
            df = pd.read_csv("../predictions/cv_results.csv")
            data = {
                "model": [self.model],
                "fold_0": [self.acc[0][0]],
                "fold_1": [self.acc[0][1]],
                "fold_2": [self.acc[0][2]],
                "fold_3": [self.acc[0][3]],
                "fold_4": [self.acc[0][4]],
                "accuracy": [np.mean(self.acc)]
            }
            df = pd.concat([df,pd.DataFrame(data=data)],axis=0)
            df.to_csv("../predictions/cv_results.csv",index=False)

        else:
            data = {
                "model": [self.model],
                "fold_0": [self.acc[0][0]],
                "fold_1": [self.acc[0][1]],
                "fold_2": [self.acc[0][2]],
                "fold_3": [self.acc[0][3]],
                "fold_4": [self.acc[0][4]],
                "accuracy": [np.mean(self.acc)]
            }
            
            df = pd.DataFrame(data=data)
            df.to_csv("../predictions/cv_results.csv",index=False)
