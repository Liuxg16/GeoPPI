import numpy as np
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pickle
import scipy, os, time
import os.path as path
import sys, random
seed=33
# default setting of GBT
lr = 0.001
subs = 0.4 
st = 3
random.seed(seed)
np.random.seed(seed)


def main(datadir,X, Y,train_index, test_index):
    workdir = 'temp/'
    idx = np.load(datadir+'sorted_idx.npy')
    depth_list = [4,6,8]
    hidsize_list  = [100,120,140]
    nest_list= [30000,40000,50000]


    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    valN = len(Y_train)//10
    xval = X_train[:valN]
    yval = Y_train[:valN]
    xtrain = X_train[valN:]
    ytrain = Y_train[valN:]

    #hyper-parameter tuning for each CV cycle
    maxr = -1
    for depth in depth_list:
        for n_est in nest_list:
            for nfeat in hidsize_list:
                xtrain1 = xtrain[:,idx[0:nfeat]]
                xval1 = xval[:,idx[0:nfeat]]
                forest1 =GradientBoostingRegressor(n_estimators=n_est,max_features="sqrt",learning_rate=lr,max_depth=depth,min_samples_split= st,subsample=subs)
                forest1.fit(xtrain1, ytrain)
                yyval = forest1.predict(xval1)
                pr = scipy.stats.pearsonr(yval,yyval)[0]
                if maxr<pr:
                    besthyper = [depth,n_est,nfeat]
                    maxr = pr

    depth,n_est,nfeat = besthyper
    n_num = X.shape[0]
    result = np.zeros(n_num)
    X_train = X_train[:,idx[0:nfeat]]
    X_test = X_test[:,idx[0:nfeat]]
    model = GradientBoostingRegressor(n_estimators=n_est, max_features = "sqrt",learning_rate=lr,max_depth=depth, min_samples_split= st, subsample=subs).fit(X_train, Y_train)
    result[test_index] = model.predict(X_test)
    np.save(workdir+'cv-{}'.format(fold), result)

# MAIN
if __name__ == '__main__':
    datadir = sys.argv[1]
    workdir = 'temp/'
    with open(datadir+'divided-folds.pkl', 'rb') as infile:
        folds = pickle.load(infile)
    foldN = len(folds)

    X = np.load("{}/X.npy".format(datadir))
    Y = np.load("{}/Y.npy".format(datadir))
    Y = Y.astype(float)

    if len(sys.argv)==2:
        if path.exists('./{}'.format(workdir)):
            os.system('rm -r {}'.format(workdir))

        os.system('mkdir {}'.format(workdir))
        print('----------Start five-fold split-by-structure cross validation-------------')
        print('--------------------It will take around 30 minutes---------------------------')
        for i in range(foldN):
            if i <foldN-1:
                os.system('python -u  sscv.py {} {} & '.format(datadir, i))
            else:
                os.system('python -u   sscv.py  {}  {}  '.format(datadir,i))

        time.sleep(30)
        os.system('python -u  sscv.py  {} test test'.format(datadir))

    elif len(sys.argv)==3:
        fold = int(sys.argv[2])
        train_index, test_index  = folds[fold]
        main(datadir,X,Y,train_index, test_index)
    elif len(sys.argv)==4:
        result = 0
        ids= []
        print('----------Correlation in each fold-------------')
        for fold in range(foldN):
            test_index = folds[fold][1]
            res = np.load(workdir+'cv-{}.npy'.format(fold))
            pr = scipy.stats.pearsonr(Y[test_index],res[test_index])
            print(fold, pr[0])
            result = res+result

        print('------------Overall correlation----------------')
        pearsonr = scipy.stats.pearsonr(Y,result)
        print(pearsonr[0])
        print('--------------------End------------------------')


