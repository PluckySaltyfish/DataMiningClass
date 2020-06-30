import pandas as pd
import numpy as np
import fm
from time import time
from sklearn.model_selection import train_test_split
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
random_state = np.random.RandomState(42)
df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc',
              'ABOD', 'CBLOF', 'FB', 'HBOS', 'IForest', 'KNN', 'LOF', 'MCD',
              'OCSVM', 'PCA']
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)
file_lst = fm.get_filelist('../../data/spambase/benchmarks/',[])
file_lst.sort()
for file in file_lst:
    print('processing file '+ file[-8:-4])
    print('----------')
    df = pd.read_csv(file)
    x = df.drop(['ground.truth','point.id','motherset','origin','original.label'],axis = 1).values
    y = df['ground.truth'].values
    y = [0 if i == 'nominal' else 1 for i in y]

    outliers_fraction = min(np.count_nonzero(y) / len(y),0.5)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)
    
    roc_list = [file[-8:-4], x.shape[0], x.shape[1], outliers_percentage]
    prn_list = [file[-8:-4], x.shape[0], x.shape[1], outliers_percentage]
    time_list = [file[-8:-4], x.shape[0], x.shape[1], outliers_percentage]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,random_state=random_state)
    x_train_norm, x_test_norm = standardizer(x_train, x_test)
    
    classifiers = {'Angle-based Outlier Detector (ABOD)': ABOD(
        contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor': CBLOF(
            contamination=outliers_fraction, check_estimator=False,
            random_state=random_state),
        'Feature Bagging': FeatureBagging(contamination=outliers_fraction,
                                          random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS(
            contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,
                                    random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Local Outlier Factor (LOF)': LOF(
            contamination=outliers_fraction),
        'Minimum Covariance Determinant (MCD)': MCD(
            contamination=outliers_fraction, random_state=random_state),
        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
        'Principal Component Analysis (PCA)': PCA(
            contamination=outliers_fraction, random_state=random_state),
    }
    
    for clf_name, clf in classifiers.items():
        try:
            t0 = time()
            clf.fit(x_train_norm)
            test_scores = clf.decision_function(x_test_norm)
            t1 = time()
            duration = round(t1 - t0, ndigits=4)
            roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
            prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
        except Exception as e:
            roc = 0
            prn = 0
            duration = 0
        print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
                  'execution time: {duration}s'.format(
                clf_name=clf_name, roc=roc, prn=prn, duration=duration))
        time_list.append(duration)
        roc_list.append(roc)
        prn_list.append(prn)

    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)

    temp_df = pd.DataFrame(roc_list).transpose()
    temp_df.columns = df_columns
    roc_df = pd.concat([roc_df, temp_df], axis=0)

    temp_df = pd.DataFrame(prn_list).transpose()
    temp_df.columns = df_columns
    prn_df = pd.concat([prn_df, temp_df], axis=0)
time_df.to_csv("spambase-time.csv",index = False)
roc_df.to_csv("spambase-roc.csv",index = False)
prn_df.to_csv("spambase-prn.csv",index = False)
print("All file saved")