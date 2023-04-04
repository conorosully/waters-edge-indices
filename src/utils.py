#A list of functions that are used in the main script

#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
from PIL import Image
from osgeo import gdal
from skimage import feature

from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

import os
import glob
import random

import xarray as xr
import spyndex
import shap


class data_processor: 

    def __init__(self,channels,standards,indices):
        self.channels = channels
        self.standards = standards
        self.indices = indices

    def get_rgb(self,img):
        """Return normalized RGB channels from sentinal image"""
        
        rgb_img = img[:, :, [3,2,1]]
        rgb_normalize = np.clip(rgb_img/10000, 0, 0.3)/0.3
        
        return rgb_normalize

    def load_test(self,path):
        """Returns sentinal image, rgb image and label"""
        
        img = gdal.Open(path).ReadAsArray()
        stack_img = np.stack(img, axis=-1)
        rgb_img = self.get_rgb(stack_img)
        
        label_path = path.replace("images","labels").replace("image","label")

        # Water = 1, Land = 0
        label = gdal.Open(label_path).ReadAsArray()
        
        return stack_img, rgb_img, label
    

    def add_indicies(self,img):

        """Add indices to image"""

        da = xr.DataArray(
            img,
            dims = ("band","x","y"),
            coords = {"band": self.channels}
        )

        params = {standard: da.sel(band = channel) for standard,channel in zip(self.standards,self.channels)}

        idx = spyndex.computeIndex(
            index = self.indices,
            params = params)

        img = np.array(np.vstack((img,idx)))

        return img 
    
    def get_data(self,paths):

        """Load all images and add labels"""

        # Load all images
        input = []
        rgb = []
        labels = []

        for path in paths:
        
            img, rgb_img, label = self.load_test(path)
            
            # transpose and scale image
            img = np.clip(img.transpose(2,0,1)/10000,0,1)

            # add index
            img = self.add_indicies(img)

            input.append(img)
            labels.append(label)
            rgb.append(rgb_img)
        
        return input, rgb, labels
    
    def img_corr(self,img,label):
        """Returns correlation between two images"""
        img = np.array(img)
        label = np.array(label)
        
        img = img.flatten()
        label = label.flatten()

        corr = np.corrcoef(img,label)[0,1]
        corr = np.abs(corr)
        
        return corr

    def img_mutual_info(self,img,label):
        """Returns mutual information between a band and a label"""
        img = np.array(img)
        label = np.array(label)
        
        img = img.flatten().reshape(-1, 1)
        label = label.flatten()

        mutual_info = mutual_info_classif(img,label)[0]
        
        return mutual_info

    def avg_intesity(self,img,label):
        """Return average intersection intensity of image and label"""
        img = np.array(img)
        label = np.array(label)

        img = img.flatten()
        label = label.flatten()

        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)

        int_water =  np.mean(img[label==1])
        int_land = np.mean(img[label==0])
        diff = int_land - int_water
        
        return diff    
    
    def get_xgb_dataset(self,input,labels):
        """Return dataset for xgboost"""

        xgb_dataset = {feature:[] for feature in self.channels + self.indices}

        for i in range(len(input)):
            for j,feature in enumerate(self.channels + self.indices):
                xgb_dataset[feature].extend(input[i][j].flatten())

        xgb_dataset = pd.DataFrame(xgb_dataset)

        xgb_dataset['lables'] = np.array(labels).flatten()

        return xgb_dataset
    
class data_visualizer:

    def __init__(self,channels,standards,indices,df_metrics):
        self.channels = channels
        self.standards = standards
        self.indices = indices
        self.df_metrics = df_metrics
        self.fig_path = '/Users/conorosullivan/Google Drive/My Drive/UCD/research/Indices/Figures/{}.png'

    def plot_indices(self,img,rgb_img,label,save=None):
        """Plot indices of image"""
    
        n_channels = len(self.channels)

        # plot indices
        fig, axes = plt.subplots(2,5,figsize=(20,8))
        fig.set_facecolor('white')

        axes = axes.flatten()
        for i,ax in enumerate(axes):

            if i == 0:
                ax.imshow(rgb_img)
                ax.set_title('RGB',fontsize=20)
            elif i == 1:
                ax.imshow(label, cmap='gray')
                ax.set_title('Ground Truth \n Mask',fontsize=20)
            elif i == 2:
                ax.imshow(img[7],cmap='gray')
                ax.set_title('NIR',fontsize=20)
            else:
                ax.imshow(img[i-3 +n_channels],cmap='gray')
                title = str(self.indices[i-3])
                ax.set_title(title,fontsize=20)
            
            ax.set_xticks([])
            ax.set_yticks([])

        if save:
            plt.savefig(self.fig_path.format(save), dpi=300, bbox_inches='tight')

    def boxplot(self,metric, ylabel="", order=False, save=None):
        """Plot boxplot of metric for each band"""

        fig = plt.figure(figsize=(12,3))
        fig.set_facecolor('white')

        channels_ = self.channels + self.indices

        if order:
            df = self.df_metrics[['band',metric]].apply('abs').groupby('band').mean().reset_index()
            df = df.sort_values(by=metric,ascending=False)
            order = df['band'].values

            sns.boxplot(x="band", y=metric, data=self.df_metrics,order=order, color='C0')
        
            #order channels
            channels_ = [channels_[i-1] for i in order]
        else:
            sns.boxplot(x="band", y=metric, data=self.df_metrics, color='C0')
            
        plt.xlabel("")
        plt.ylabel(ylabel,size=20)   
        channels_ = [i.replace(" "," \n") if len(i) >=12 else i for i in channels_]
        plt.xticks(np.arange(0,len(channels_)),channels_,rotation=90, size=15)
        plt.yticks(size=15)

        if save != None:
            plt.savefig(self.fig_path.format(save), dpi=300, bbox_inches='tight')


class model_eval:
    def __init__(self,model,df):
        self.model = model
        self.df = df

        self.features = model.feature_names_in_
        self.X = df[self.features]
        self.y = df['lables']

        self.fig_path = '/Users/conorosullivan/Google Drive/My Drive/UCD/research/Indices/Figures/{}.png'
    
    def calc_acc(self):
    # Get model accuracy
        preds = self.model.predict(self.X)
        acc = np.mean(preds == self.y)

        print("Model Accuracy: ",acc)
        return acc
    
    def get_shap_values(self,X,n=None):
        """Get shap values for model"""
        
        explainer = shap.Explainer(self.model)

        if n:
            shap_values = explainer(X.sample(n))
        else:
            shap_values = explainer(X)

        return shap_values
    

    def plot_importance(self,importance, ylabel,order=False, save=None):
        """Plot model importance score"""

        if order:
            index = np.argsort(importance)[::-1]
        else:
            index = np.arange(len(importance))

        labels = np.array(self.features)[index]
        importance = importance[index]

        #Plot feature importance
        plt.figure(figsize=(10,3))
        plt.bar(range(len(importance)), importance)
        labels = [i.replace(" "," \n") if len(i) >=12 else i for i in labels]
        plt.xticks(range(len(importance)), labels, rotation='vertical',size=15)
        plt.yticks(size=15)
        plt.ylabel(ylabel,size=20)

        if save:
            plt.savefig(self.fig_path.format(save), dpi=300, bbox_inches='tight')

    def plot_feature_importance(self,order=False, save=None,to_return=False):

        """Plot feature importance"""

        #importance = self.model.feature_importances_
        importance = permutation_importance(self.model, self.X, self.y,n_jobs=-1).importances_mean  

        if to_return:
            return importance
        else: 
            self.plot_importance(importance,"Permutation Feature \nImportance",order,save=save)
    

    def plot_shap_importance(self,order=False,n=None,save=None):

        """Plot shap importance"""
        shap_values = self.get_shap_values(self.X,n)

        importance = np.abs(shap_values.values).mean(0)
        self.plot_importance(importance,"Mean SHAP", order,save=save)