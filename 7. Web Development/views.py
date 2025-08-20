from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
#from .forms import *
from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
ListView,DetailView,
CreateView,DeleteView,
UpdateView)
from . import models
from .forms import *
from django.core.files.storage import FileSystemStorage
#from topicApp.Topicfun import Topic
#from ckdApp.funckd import ckd
#from sklearn.tree import export_graphviz #plot tree
#from sklearn.metrics import roc_curve, auc #for model evaluation
#from sklearn.metrics import classification_report #for model evaluation
##from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)

import time
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#from sklearn.model_selection import train_test_split
#from sklearn.decomposition import PCA
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
#import eli5 #for purmutation importance
#from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
#from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproduc
class dataUploadView(View):
    form_class = ckdForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        #print('inside form')
        if form.is_valid():
            form.save()
            Operation_FileAccessed_input= request.POST.get('FileAccessed_or_not')
            Operation_FileModified_input=request.POST.get('FileModified_or_not')
            Operation_FileUploaded_input=request.POST.get('FileUploaded_or_not')
            Operation_FileDownloaded_input=request.POST.get('FileDownloaded_or_not')
            Operation_MoveToDeletedItems_input=request.POST.get('MoveToDeletedItems')
            IsRiskyHour_input=request.POST.get('IsRiskyHour_or_not')
            Operation_UserLoginFailed_input=request.POST.get('UserLoginFailed_or_not')
            GeoLocation_IND_input=request.POST.get('GeoLocation_IND_or_not')
            Unknow_ResultStatus_input=request.POST.get('Unknow_ResultStatus_or_not')
            #print (data)
            #dataset1=pd.read_csv("prep.csv",index_col=None)
            dicc={'yes':1,'no':0}
            filename = 'Finalized_RandomForestClassifier_model.sav'
            classifier = pickle.load(open(filename, 'rb'))

            data = np.array([Operation_FileAccessed_input,Operation_FileModified_input,Operation_FileUploaded_input,
            Operation_FileDownloaded_input,Operation_MoveToDeletedItems_input,IsRiskyHour_input,
            Operation_UserLoginFailed_input,GeoLocation_IND_input,Unknow_ResultStatus_input])
            #sc = StandardScaler()
            #data = sc.fit_transform(data.reshape(-1,1))
            out=classifier.predict(data.reshape(1,-1))
# providing an index
            #ser = pd.DataFrame(data, index =['bgr','bu','sc','pcv','wbc'])

            #ss=ser.T.squeeze()
#data_for_prediction = X_test1.iloc[0,:].astype(float)

#data_for_prediction =obj.pca(np.array(data_for_prediction),y_test)
            #obj=ckd()
            ##plt.savefig("static/force_plot.png",dpi=150, bbox_inches='tight')







            return render(request, "succ_msg.html", {'data_fileAccessed':Operation_FileAccessed_input,
                                                     'data_fileModified':Operation_FileModified_input,
                                                     'data_fileUploaded':Operation_FileUploaded_input,
                                                     'data_fileDownloaded':Operation_FileDownloaded_input,
                                                     'data_deletedItems':Operation_MoveToDeletedItems_input,
                                                     'data_isRiskyHour':IsRiskyHour_input,
                                                     'data_userLoginFailed':Operation_UserLoginFailed_input,
                                                     'data_geoLocation':GeoLocation_IND_input,
                                                     'data_resultStatus':Unknow_ResultStatus_input,
                                                        'out':out})


        else:
            return redirect(self.failure_url)
