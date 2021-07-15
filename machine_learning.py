# Basic Libraries
import pandas as pd
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split
from math import sin, cos, sqrt, atan2, radians
from sklearn import metrics
import seaborn as sb
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Importing the Packages:
import optuna
from sklearn import datasets, model_selection, ensemble, linear_model

import streamlit as st
import matplotlib.pyplot as plt 

import numpy as np
import itertools
import time
from utils import *
import pickle
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold



def calculate_distance(dataframe):
    temp = pd.DataFrame()
    # approximate radius of earth in km
    R = 6373.0
    # calculating and storing distance as a record in a new column in main dataframe
    temp["lat1"] = np.deg2rad(dataframe["customer_lat"])
    temp["lon1"] = np.deg2rad(dataframe["customer_lng"])
    temp["lat2"] = np.deg2rad(dataframe["seller_lat"])
    temp["lon2"] = np.deg2rad(dataframe["seller_lng"])
    
    temp["dlon"] = temp["lon2"] - temp["lon1"]
    temp["dlat"] = temp["lat2"] - temp["lat1"]
    
    temp ["a"] = np.square(np.sin(temp["dlat"]/2.0)) + np.cos(temp["lat1"]) * np.cos(temp["lat2"]) * np.square(np.sin(temp["dlon"]/2.0))
    
    temp["c"] = 2.0 * np.arctan2(np.sqrt(temp.a),np.sqrt(1.0-temp.a))
    dataframe["distance"] = temp["c"]*R

    return dataframe

def result_confusion_matrix(y,y_pred,labels,title):    
    confusion_m = confusion_matrix(y,y_pred,labels = labels )
 
    df_cm = pd.DataFrame(confusion_m, index = [i for i in labels],
                      columns = [i for i in labels])
    sb.set(font_scale = 0.1)
    
    fig, axes = plt.subplots(figsize=(2, 2))
    #plt.yticks(fontsize="5",va="center",ha = "center")
    #plt.xticks(fontsize="5",va="center",ha = "center")
    sb.heatmap(df_cm, 
               cmap='YlGnBu',
               annot=True,
               fmt=".0f",
               annot_kws={'size':6},
               linewidths=.01,
               xticklabels="auto", 
               yticklabels="auto",
               linecolor='black',
               square=True,
               cbar=False)#,ax=axes)
               
    axes.axhline(y=0, color='black',linewidth=0.4)
    axes.axhline(y=len(labels), color='black',linewidth=1.0)
    axes.axvline(x=len(labels), color='black',linewidth= 1)
    axes.axvline(x=0, color='black',linewidth= 0.4)
    
    axes.set_xlabel("Predicted Delivery Days", fontsize = 7)
    axes.set_ylabel("Actual Delivery Days", fontsize = 7)
    
    #plt.title(title, fontsize =8)
    fig.suptitle(title, fontsize=8,ha="left",va="top")
    _, xlabels = plt.xticks()
    axes.set_xticklabels(labels, size=6, ha="center",va="center")
    _, ylabels = plt.yticks()
    axes.set_yticklabels(labels, size=6, ha="center",va="center")
    fig.tight_layout()
    
    recall = np.round(np.mean(np.diag(confusion_m) / np.sum(confusion_m, axis = 1)),2)
    precision = np.round(np.mean(np.diag(confusion_m) / np.sum(confusion_m, axis = 0)),2)
    
    return(fig,recall,precision)

def error_matrix(y,y_pred,labels,Title):

    dict = {'Metric':['Accuracy','Recall', 'Precision'],
            Title:[np.round(accuracy_score(y,y_pred),2), result_confusion_matrix(y,y_pred,labels,Title)[1], result_confusion_matrix(y,y_pred,labels,Title)[2]],
           }
           
    errors = pd.DataFrame(dict)
    
    return (errors.round(2))

    
def logistic_regression(X_train, X_test, y_train, y_test):
    #log_reg = LogisticRegression(random_state=0,solver = 'newton-cg',max_iter=50000)
    #log_reg.fit(X_train, y_train)
    filename = 'logistic_regression.sav'
    #pickle.dump(log_reg, open(filename, 'wb'))
    log_reg = pickle.load(open(filename, 'rb'))
    y_train_pred = log_reg.predict(X_train)
    y_test_pred = log_reg.predict(X_test)
    
    return (y_train_pred, y_test_pred)
       
def randomforest_classification(X_train, X_test, y_train, y_test):
    '''
    def objective(trial):

        rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 1000)
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        min_samples_split = trial.suggest_int("min_samples_split",2,100)
        min_samples_leaf = trial.suggest_int("min_samples_leaf",1,20)
        
        reg_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_n_estimators, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
    
        # Step 3: Scoring method:
        score = model_selection.cross_val_score(reg_obj, X_train, y_train, n_jobs=-1, cv=5)
        accuracy = score.mean()
        return accuracy

    # Step 4: Running it
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=80, n_jobs=-1)
    
    print(f"The best trial is : \n{study.best_trial}")
    st.write("The best trial is :",study.best_trial)
    st.write(" ")
    st.write(study.best_params)
    
    clf = RandomForestClassifier(max_depth=study.best_params['rf_max_depth'],
                                 n_estimators=study.best_params['rf_n_estimators'],
                                 min_samples_split = study.best_params['min_samples_split'],
                                 min_samples_leaf = study.best_params['min_samples_leaf'],
                                 random_state=0)#,solver = 'newton-cg',max_iter=500)
    clf.fit(X_train, y_train)
    '''
    filename = 'random_forest_classifcation.sav'
    #pickle.dump(clf, open(filename, 'wb'))
    clf = pickle.load(open(filename, 'rb'))
    
    y_train_pred = clf.predict(X_train)
    y_test_pred =  clf.predict(X_test)
    
    return (y_train_pred, y_test_pred)

#def deep_learning_model(X_train, X_test, y_train, y_test):
#    '''
#    y_train = y_train.copy()
#    def objective(trial):
#
#        node_layer1 = trial.suggest_int("node_layer1", 2, 15)
#        node_layer2 = trial.suggest_int("node_layer2", 2, 15)
#        print ((len(list(X_train))))
#        print (y_test)
#              
#        
#        network = models.Sequential()
#        network.add(layers.Dense(node_layer1, activation='relu', input_shape=(len(list(X_train)),)))
#        network.add(layers.Dense(node_layer2, activation='relu'))
#        network.add(layers.Dense(len(y_test[list(y_test)[0]].unique()), activation='softmax'))
#        network.compile(optimizer='rmsprop',
#                        loss='categorical_crossentropy',
#                        metrics=['accuracy'])
#                        
#        le = LabelEncoder()
#        label_model = le.fit(y_train)
#        y_train = le.transform(y_train.copy())          
#        y_train = to_categorical(y_train.copy())
#        
#        # Fit the neural network
#        network.fit(X_train, y_train, epochs=80, batch_size= 40)# batch_size_trial)#40)
#        train_loss, train_acc = network.evaluate(X_train, y_train)
#        return train_acc
#        
#    study = optuna.create_study(direction="maximize")
#    study.optimize(objective, n_trials=20)#, timeout=300)
#
#    st.write(study.best_params['node_layer1'],study.best_params['node_layer2'])
#    #st.write(type(study.best_params['node_layer1']))
#    #network_best = models.Sequential()
#    #network_best.add(layers.Dense(3, activation='relu', input_shape=(len(list(X_train)),)))
#    #network_best.add(layers.Dense(12, activation='relu'))    
#    '''
#    network_best = models.Sequential()
#    #network_best.add(layers.Dense(study.best_params['node_layer1'], activation='relu', input_shape=(len(list(X_train)),)))
#    #network_best.add(layers.Dense(study.best_params['node_layer2'], activation='relu'))
#    network_best.add(layers.Dense(3, activation='relu', input_shape=(len(list(X_train)),)))
#    network_best.add(layers.Dense(14, activation='relu')) 
#    network_best.add(layers.Dense(len(y_train[list(y_train)[0]].unique()), activation='softmax'))
#  
#    network_best.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#
#    le = LabelEncoder()
#    label_model = le.fit(y_train)
#            
#    y_train = le.transform(y_train.copy())  
#    y_test  = le.transform(y_test.copy())    
#        
#    y_train = to_categorical(y_train.copy())
#    y_test = to_categorical(y_test.copy())

#    network_best.fit(X_train, y_train, epochs=20, batch_size= 40)
#    
#    network_best.save("deep_learning_model_4class")
#    
#    new_model = tf.keras.models.load_model("deep_learning_model_4class_final")
#    print(network_best.summary())
#    print(new_model.summary())
#    
#    #y_train_pred = network_best.predict(X_train)
#    #y_test_pred = network_best.predict(X_test)
#
#    y_train_pred = new_model.predict(X_train)
#    y_test_pred  = new_model.predict(X_test)
#
#    ##st.write("y_train_pred",y_train_pred)
#
#    ##y_classes = [np.argmax(y, axis=None, out=None) for y in y_train_pred]
#    ##st.write("y_classes",y_classes)
#    
#    y_train_classes = tf.argmax(y_train_pred,axis=1)
#    y_test_classes  = tf.argmax(y_test_pred,axis=1)
#    
#    y_classes_train_pred = le.inverse_transform(y_train_classes)
#    y_classes_test_pred  = le.inverse_transform(y_test_classes)
#    
#    return (y_classes_train_pred, y_classes_test_pred)
    
    
def result_ml_model(X_train, X_test, y_train, y_test,labels,model_name):    
    st.subheader(model_name)
    progress_bar = st.progress(0)
    progress_bar.progress(20)
    
    with st.spinner(text='Please be Patient, Good Things Take Time...'):
        if (model_name == "Logistic Regression"):
            y_train_pred, y_test_pred = logistic_regression(X_train, X_test, y_train, y_test)
            
        if (model_name == "Random Forest Classifier"):
            y_train_pred, y_test_pred = randomforest_classification(X_train, X_test, y_train, y_test)
            
#        if (model_name == "Deep Learning"):
#            y_train_pred, y_test_pred = deep_learning_model(X_train, X_test, y_train, y_test)
#            #y_train_pred, y_test_pred = deep_learning_model(X_train, X_test, y_train, y_test)
    
    
    progress_bar.progress(50)
    
    col1,col2,col3 = st.beta_columns(3)
    col1.write(result_confusion_matrix(y_train,y_train_pred,labels,"Train")[0])
    progress_bar.progress(70)
    
    col2.write(result_confusion_matrix(y_test,y_test_pred,labels,"Test")[0])
    progress_bar.progress(80)
    
    test_error_matrix  = error_matrix(y_test,y_test_pred,labels,"Test")
    train_error_matrix = error_matrix(y_train,y_train_pred,labels,"Train")
    
    error_table = train_error_matrix.merge(test_error_matrix,left_on="Metric",right_on="Metric").round(2)
    col3.write("")
    col3.write("")
    col3.write("")
    col3.write("")
    col3.subheader("Error Metrics")
    col3.write(plotly_table (error_table,list(error_table),width = 350, height = 100))
    progress_bar.progress(90)
    
    train_accuracy = train_error_matrix[train_error_matrix["Metric"]=="Accuracy"]["Train"][0]
    test_accuracy  = test_error_matrix[test_error_matrix["Metric"]=="Accuracy"]["Test"][0]
    if (train_accuracy >= 0.6 and test_accuracy >= 0.6):   
        if (test_accuracy/train_accuracy <= 0.8):
            col3.warning("Model is over-fit")
        else:
            col3.info("Model is a good fit")
    else:
        if (test_accuracy/train_accuracy <= 0.8):
            col3.warning("Model is over-fit with bad accuracies")
        else:
            col3.warning("Model is under-fit")
    
    progress_bar.progress(100)
    st.markdown("-----")

def ml_tab(dataframe):
    
    #dataframe = calculate_distance(dataframe)
    st.sidebar.markdown("""---""")
    #distancecb = st.sidebar.checkbox("Include 'distance' as a parameter",value = True)
    
    outlierCB = st.sidebar.checkbox("Remove Outliers",value = True)
    
    dataframe = calculate_distance(dataframe)
    predictors = ["Price","Freight Charges","Review Score","Product Weight (gm)","Product Category","Payment Value","Product size","Estimated Delivery Days","Distance"]
    
    vals_list_temp = st.sidebar.multiselect("Predictors", predictors, default=predictors)
    col_names = read_json('col_names.json')
    col_name_rev = dict((v,k) for k,v in col_names.items())
    
    vals_list = []
    for i in vals_list_temp:
        vals_list.append(col_name_rev[i])
        
    
    #st.write("vals in select bar",vals_list)
    vals_to_remove_outlier = remove_list_parameters(vals_list.copy(),["review_score","Types of products"])
    vals_to_remove_outlier = vals_to_remove_outlier.copy() + ["delivery_days"]
    vals_consider = vals_list.copy()+ ["delivery_days_binned"]
    #st.write("vals_remove_outlier:",vals_to_remove_outlier)
    #st.write("vals_consider:",vals_consider)
    
    if outlierCB:

        dataframe = remove_outlier_dataframe(dataframe, vals_to_remove_outlier)
        #st.write(len(dataframe))
    else:
        dataframe = dataframe.copy()
        #st.write(len(dataframe))
        
    #st.write(list(dataframe))
    #st.dataframe(dataframe)
    
    bins = [0,5,10,20,30]
    dataframe["delivery_days_binned"] = pd.cut(dataframe["delivery_days"],bins = bins).astype("str")
    dataframe["delivery_days_binned"] = dataframe["delivery_days_binned"].fillna(">30")
    #st.write(dataframe["delivery_days_binned"].value_counts())
    #st.write(dataframe.infer_objects().dtypes)
    #(0.0, 5.0] #(5.0, 10.0] #(10.0, 20.0] #(20.0, 30.0]
     
    dataframe["delivery_days_binned"] = dataframe["delivery_days_binned"].replace('(0, 5]',"1-5").copy()
    dataframe["delivery_days_binned"] = dataframe["delivery_days_binned"].replace('(5, 10]',"6-10").copy()
    dataframe["delivery_days_binned"] = dataframe["delivery_days_binned"].replace('(10, 20]',"11-20").copy()
    dataframe["delivery_days_binned"] = dataframe["delivery_days_binned"].replace('(20, 30]',"21-30").copy()
    dataframe["delivery_days_binned"] = dataframe["delivery_days_binned"].replace('nan',">30").copy()
    #st.write(dataframe["delivery_days_binned"].value_counts())
    
    dataframe = dataframe[vals_consider].copy()
    #st.write (dataframe.shape)
    #st.write(list(dataframe))
    
    products = pd.get_dummies(dataframe["Types of products"])
    dataframe = pd.concat([dataframe.copy(),products], axis=1)
    #st.write (dataframe.shape)

    review  = pd.get_dummies(dataframe["review_score"])
    dataframe = pd.concat([dataframe.copy(),review], axis=1)
    #st.write (dataframe.shape)

    dataframe = dataframe.drop(columns=['Types of products', 'review_score']).copy()
    dataframe = dataframe.drop_duplicates().copy()
    dataframe = dataframe.dropna().copy()
        
    y = pd.DataFrame(dataframe["delivery_days_binned"]).copy()
    X = dataframe[dataframe.columns.drop("delivery_days_binned")].copy()    
    X = X.T.drop_duplicates().T.copy()
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    labels = list(y_test["delivery_days_binned"].unique())
    #st.write(labels)
    #st.write(y_test["delivery_days_binned"].value_counts())
    if len(labels)==4:
        labels = ["1-5","6-10","11-20","21-30"] #labels.sort()
    else:
        labels = ["1-5","6-10","11-20","21-30",">30"]
    
    model_types = ["Logistic Regression","Random Forest Classifier"]#,"Deep Learning"]
    select_model = st.multiselect("Select Machine Learning Models", model_types, default=model_types)

    if "Logistic Regression" in select_model:
        result_ml_model(X_train, X_test, y_train, y_test,labels,"Logistic Regression")
        
    if "Random Forest Classifier" in select_model:
        result_ml_model(X_train, X_test, y_train, y_test,labels,"Random Forest Classifier")

#    if "Deep Learning" in select_model:
#        result_ml_model(X_train, X_test, y_train, y_test,labels,"Deep Learning")        

    st.balloons()
    
    
#rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=2652124)
        #count=1
        #val_accuracy = []
        #for train_index, test_index in rkf.split(X_train):
        #    X_train_val, X_test_val = X_train.iloc[train_index.tolist()], X_train.iloc[test_index.tolist()]
        #    y_train_val, y_test_val = y_train.iloc[train_index.tolist()], y_train.iloc[test_index.tolist()]
        
        #    network = models.Sequential()
        #    network.add(layers.Dense(node_layer1, activation='relu', input_shape=(len(list(X_train_val)),)))
        #    network.add(layers.Dense(node_layer2, activation='relu'))
        #    network.add(layers.Dense(len(y_train_val[list(y_train_val)[0]].unique()), activation='softmax'))
          
        #    network.compile(optimizer='rmsprop',
        #                    loss='categorical_crossentropy',
        #                    metrics=['accuracy'])
        #                
        #    le = LabelEncoder()
        #    label_model = le.fit(y_train_val)
        #    
        #    y_train_val = le.transform(y_train_val.copy())          
        #    y_train_val = to_categorical(y_train_val.copy())
        #    
        #    # Fit the neural network
        #    network.fit(X_train_val, y_train_val, epochs=80, batch_size= 40)# batch_size_trial)#40)

        #    train_loss, train_acc = network.evaluate(X_train_val, y_train_val)
        #    #test_loss, test_acc = network.evaluate(X_test_val, y_test_val)
        #    #st.write(str(count)+': Test Accuracy: '+str(test_acc)+ '\nTest Loss: '+str(test_loss))
        #    val_accuracy.append(train_acc)
        #    count+=1
        
        #return np.sqrt(np.mean(val_accuracy) ** 2 + np.std(val_accuracy) ** 2)
