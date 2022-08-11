import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import PyPluMA

class HoltWintersPlugin:
    def input(self, filename):
       self.parameters = dict()
       infile = open(filename, 'r')
       for line in infile:
           contents = line.strip().split('\t')
           self.parameters[contents[0]] = contents[1]
       if (self.parameters["seasonal"] == "false"):
          self.seasonal = False
          self.dataset = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["csvfile"])
          self.train = self.dataset[0:int(self.parameters["trainend"])]
          self.test = self.dataset[int(self.parameters["trainend"])+1:]
       else:
          self.seasonal = True
          self.df = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["csvfile"])
          trainend = int(self.parameters["trainend"])
          indexfile = open(PyPluMA.prefix()+"/"+self.parameters["indexfile"], 'r')
          indices = []
          for line in indexfile:
              indices.append(int(line.strip()))
          self.train, self.test = self.df.iloc[:trainend,indices], self.df.iloc[trainend+1:,indices]
          print(dir(self.train))
          #model = ExponentialSmoothing(self.train.Temperature, seasonal='add', seasonal_periods=int(self.parameters["seasonalperiods"])).fit()
          self.model = ExponentialSmoothing(self.train.Temperature, seasonal='add', seasonal_periods=int(self.parameters["seasonalperiods"])).fit()
          #model = ExponentialSmoothing(self.train.__getattribute__(self.parameters["trainparam"]), seasonal='add', seasonal_periods=int(self.parameters["seasonalperiods"])).fit()

    def run(self):
        if (not self.seasonal):
           self.y_hat_avg = self.test.copy()
           self.trainparam = self.parameters["trainparam"]
           fit1 = ExponentialSmoothing(np.asarray(self.train[self.trainparam])).fit()
           self.y_hat_avg['Holt_Winter'] = fit1.forecast(len(self.test))
        else:
           self.pred = self.model.predict(start=self.test.num[int(self.parameters["teststart"])], end=self.test.num[int(self.parameters["testend"])])

    def output(self, filename):
       if (not self.seasonal):
        plt.figure(figsize=(16,8))
        plt.plot( self.train[self.trainparam], label='Train')
        plt.plot(self.test[self.trainparam], label='Test')
        plt.plot(self.y_hat_avg['Holt_Winter'], label='Holt_Winter')
        plt.legend(loc='best')
        plt.savefig(filename)
       else:
        plt.figure(figsize=(64,8))
        plt.scatter(self.train.num, self.train.Temperature, label='Train',s=1)
        plt.scatter(self.test.num, self.test.Temperature, label='Test',s=1)
        plt.scatter(self.test.num,self.pred,s=1)
        plt.legend(loc='best')
        plt.savefig(filename)
        plt.show()
        rms = sqrt(mean_squared_error(self.test.Temperature, self.pred))
        print(rms)
