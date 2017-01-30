# usage: python classify_nodes.py nodes.npy 

import numpy as np
import pickle

from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb

def getRegionFromMap(slice_npy):
    thr = np.where(slice_npy > np.mean(slice_npy),0.,1.0)
    label_image = label(thr)
    labels = label_image.astype(int)
    regions = regionprops(labels)
    return regions

def getRegionMetricRow(fname = "nodules.npy"):
    # fname, numpy array of dimension [#slices, 1, 512, 512] containing the images
    seg = np.load(fname)
    nslices = seg.shape[0]
    
    #metrics
    totalArea = 0.
    avgArea = 0.
    maxArea = 0.
    avgEcc = 0.
    avgEquivlentDiameter = 0.
    stdEquivlentDiameter = 0.
    weightedX = 0.
    weightedY = 0.
    numNodes = 0.
    numNodesperSlice = 0.
    # crude hueristic to filter some bad segmentaitons
    # do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
    maxAllowedArea = 0.10 * 512 * 512 
    
    areas = []
    eqDiameters = []
    for slicen in range(nslices):
        regions = getRegionFromMap(seg[slicen,0,:,:])
        for region in regions:
            if region.area > maxAllowedArea:
                continue
            totalArea += region.area
            areas.append(region.area)
            avgEcc += region.eccentricity
            avgEquivlentDiameter += region.equivalent_diameter
            eqDiameters.append(region.equivalent_diameter)
            weightedX += region.centroid[0]*region.area
            weightedY += region.centroid[1]*region.area
            numNodes += 1
            
    weightedX = weightedX / totalArea 
    weightedY = weightedY / totalArea
    avgArea = totalArea / numNodes
    avgEcc = avgEcc / numNodes
    avgEquivlentDiameter = avgEquivlentDiameter / numNodes
    stdEquivlentDiameter = np.std(eqDiameters)
    
    maxArea = max(areas)
    
    
    numNodesperSlice = numNodes*1. / nslices
    
    
    return np.array([avgArea,maxArea,avgEcc,avgEquivlentDiameter,\
                     stdEquivlentDiameter, weightedX, weightedY, numNodes, numNodesperSlice])


def createFeatureDataset(nodfiles=None):
    if nodfiles == None:
        # directory of numpy arrays containing masks for nodules
        # found via unet segmentation
        noddir = "/training_set/" 
        nodfiles = glob(noddir +"*npy")
    # dict with mapping between training examples and true labels
    # the training set is the output masks from the unet segmentation
    truthdata = pickle.load(open("truthdict.pkl",'r'))
    numfeatures = 9
    feature_array = np.zeros((len(nodfiles),numfeatures))
    truth_metric = np.zeros((len(nodfiles)))
    
    for i,nodfile in enumerate(nodfiles):
        patID = nodfile.split("_")[2]
        truth_metric[i] = truthdata[int(patID)]
        feature_array[i] = getRegionMetricRow(nodfile)
    
    np.save("dataY.npy", truth_metric)
    np.save("dataX.npy", feature_array)

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


def classifyData():
    X = np.load("dataX.npy")
    Y = np.load("dataY.npy")

    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
        clf = RF(n_estimators=100, n_jobs=3)
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss",logloss(Y, y_pred))

    # All Cancer
    print "Predicting all positive"
    y_pred = np.ones(Y.shape)
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss",logloss(Y, y_pred))

    # No Cancer
    print "Predicting all negative"
    y_pred = Y*0
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss",logloss(Y, y_pred))

    # try XGBoost
    print ("XGBoost")
    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
        clf = xgb.XGBClassifier(objective="binary:logistic")
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss",logloss(Y, y_pred))

if __name__ == "__main__":
    from sys import argv  
    
    getRegionMetricRow(argv[1:])
    classifyData()
