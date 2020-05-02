import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

root = '../Assignment4/oakland-crime-statistics-2011-to-2016'
file_lst = ['records-for-' + str(x) + '.csv' for x in range(2011, 2017)]


def load_data(path, filename):
    return pd.read_csv(path + '/' + filename, keep_default_na=False, low_memory=False)


def show_col():
    return load_data(root, file_lst[0]).columns


def get_row_index(index, col, value):
    data = load_data(root, file_lst[index])
    lst = []
    entity = []
    for i in range(data.shape[0]):
        if data.loc[i, col] == value:
            lst.append(i)
            entity.append(data.loc[i].values)
    return lst, entity


def delete_row(file_index, index_lst):
    data = load_data(root, file_lst[file_index])
    data = data.drop(index=index_lst)
    data.to_csv(root + '/' + file_lst[file_index],index=False)

class col_helper():
    def __init__(self):
        self.data = []

    def grab_col(self, col):
        self.data = []
        for i in file_lst:
            self.data.append(load_data(root, i)[col])
        self.data = np.array(self.data)

    def generate_new_col(self,col1,col2,f):
        self.data = []
        for i in file_lst:
            res = f(load_data(root, i)[col1],load_data(root, i)[col2])
            self.data.append(res)
        self.data = np.array(self.data)
    def five_number(self):
        year = 2011
        for i in self.data:
            print('Year',str(year))

            print('Min:', np.min(i),end=',')
            print('Q1:', np.percentile(i, 25),end=',')
            print('Q2:', np.percentile(i, 50),end=',')
            print('Q3:', np.percentile(i, 75),end=',')
            print('Max:', np.max(i))
            year += 1

    def box(self,index,w,h,xlabel):
        col = np.array(self.data[index])
        fig = plt.figure(figsize=(w, h))
        plt.boxplot(col, notch=False, vert=False)
        plt.xlabel(xlabel)
        plt.show()
        outlier = np.percentile(col, 75) + (np.percentile(col, 75) - np.percentile(col, 25)) * 1.5
        print(outlier)

    def seg_hist(self,index,s1,s2,w,h,xlabel,ylabel):
        col = np.array(self.data[index])
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

    def normal_hist(self,index,w,h,xlabel,ylabel):
        col = np.array(self.data[index])
        fig1 = plt.figure(figsize=(w, h))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.hist(col, bins=40, facecolor="#cddc39", edgecolor="#afb42b", alpha=0.7)

    # def get_fre(self):
    #     col = self.data[self.column]
    #     print(col.value_counts())
    #
    # def normal_pie(self,w,h):
    #     col = np.array(self.data[self.column].value_counts())
    #     fig = plt.figure(figsize=(w, h))
    #     explode = (0, 0.1)
    #     colors = ['#cddc39', '#fbc02d']
    #     plt.pie(col, autopct="%.2f%%", labels=['False', 'True'],
    #             startangle=150, explode=explode, colors=colors)
    #     plt.show()
    #
    def normal_bar(self, index, w, h, n_before, n_after, xticks):
        fig = plt.figure(figsize=(w, h))
        plt.subplot(121)
        plt.title('before')
        yticks = [self.data[index].shape[0] - n_before, n_before]
        plt.bar(xticks, [self.data[index].shape[0] - n_before, n_before], facecolor='#cddc39', edgecolor='#afb42b')
        for x, y in zip(xticks, yticks):
            plt.text(x, y, '%d' % y, ha='center', va='bottom')
        plt.subplot(122)
        plt.title('After')
        yticks = [self.data[index].shape[0] - n_after, n_after]
        plt.bar(xticks, [self.data[index].shape[0] - n_after, n_after], facecolor='#cddc39', edgecolor='#afb42b')
        for x, y in zip(xticks, yticks):
            plt.text(x, y, '%d' % y, ha='center', va='bottom')
        plt.show()

    def count_none(self, none_token=''):
        res = []
        y = 2011
        for year in self.data:
            cnt = 0
            for i in year:
                if i == none_token:
                    cnt += 1
            res.append(cnt)
            print(y, ':', cnt)
            y += 1
        return res
        #
        # def fill_none(self,other_col,none_token=''):
        #     for i in range(self.data.shape[0]):
        #         if(self.data[self.column][i] == none_token):
        #             self.data.loc[i,self.column] = self.data.loc[i,other_col]
