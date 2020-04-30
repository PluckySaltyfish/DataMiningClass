import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
def load_data(path,filename):
    return pd.read_csv(path + '/' + filename,keep_default_na=False)

class country_data():
    def __init__(self,path,filename):
        self.data = load_data(path,filename);
        self.column = ''

    def select_col(self,col):
        self.column = col

    def five_number(self):
        col = np.array(self.data[self.column].values)
        print('Min:', col.min())
        print('Q1:', np.percentile(col, 25))
        print('Q2:', np.percentile(col, 50))
        print('Q3:', np.percentile(col, 75))
        print('Max:', col.max())

    def box(self,w,h,xlabel):
        col = np.array(self.data[self.column].values)
        fig = plt.figure(figsize=(w, h))
        plt.boxplot(col, notch=False, vert=False)
        plt.xlabel(xlabel)
        plt.show()
        outlier = np.percentile(col, 75) + (np.percentile(col, 75) - np.percentile(col, 25)) * 1.5
        print(outlier)

    def seg_hist(self,s1,s2,w,h,xlabel,ylabel):
        col = np.array(self.data[self.column].values)
        col.sort()
        p1 = np.searchsorted(col, s1)
        p2 = np.searchsorted(col, s2)
        c1 = col[:p1]
        c2 = col[p1:p2]
        c3 = col[p2:]
        fig1 = plt.figure(figsize=(w, h))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.subplot(131)
        plt.hist(c1, bins=40, facecolor="#cddc39", edgecolor="#afb42b", alpha=0.7)
        plt.subplot(132)
        plt.hist(c2, bins=40, facecolor="#cddc39", edgecolor="#afb42b", alpha=0.7)
        plt.subplot(133)
        plt.hist(c3, bins=40, facecolor="#cddc39", edgecolor="#afb42b", alpha=0.7)
        plt.show()

    def normal_hist(self,w,h,xlabel,ylabel):
        col = np.array(self.data[self.column].values)
        fig1 = plt.figure(figsize=(w, h))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.hist(col, bins=40, facecolor="#cddc39", edgecolor="#afb42b", alpha=0.7)

    def get_fre(self):
        col = self.data[self.column]
        print(col.value_counts())

    def normal_pie(self,w,h):
        col = np.array(self.data[self.column].value_counts())
        fig = plt.figure(figsize=(w, h))
        explode = (0, 0.1)
        colors = ['#cddc39', '#fbc02d']
        plt.pie(col, autopct="%.2f%%", labels=['False', 'True'],
                startangle=150, explode=explode, colors=colors)
        plt.show()

    def normal_bar(self,w,h,n_before,n_after,xticks):
        fig = plt.figure(figsize=(w, h))
        plt.subplot(121)
        plt.title('before')
        plt.bar(xticks,[self.data.shape[0]-n_before,n_before],facecolor='#cddc39', edgecolor='#afb42b')
        plt.subplot(122)
        plt.title('After')
        plt.bar(xticks,[self.data.shape[0]-n_after,n_after],facecolor='#cddc39', edgecolor='#afb42b')
        plt.show()

    def count_none(self):
        col = np.array(self.data[self.column].values);
        cnt = 0
        for i in col:
            if i == '':
                cnt += 1
        return cnt
