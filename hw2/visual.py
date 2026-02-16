import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Visual:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def histogram(self, style, figsize_x, figsize_y, a, b):
      with sns.axes_style(style):
        plt.figure(figsize=( figsize_x, figsize_y))
        sns.histplot(data=self.data, x=a, hue=b)
      return

    def boxplot(self, style, figsize_x, figsize_y, a, b, h, ox, oy):
      with sns.axes_style(style):
        plt.figure(figsize=( figsize_x, figsize_y))
        sns.boxplot(data=self.data, x=a, y=b, orient=h)
        plt.xlim((ox, oy))

      return
