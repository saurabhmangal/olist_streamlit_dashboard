# Basic Libraries
import streamlit as st
import pandas as pd
import seaborn as sb
import home
import plotly.graph_objects as go
import numpy as np
from utils import *

sb.set()  # set the default Seaborn style for graphics

# dataframe = pd.read_csv("ExportDataFrame.csv", header=0)


@st.cache
def uni_plot(series, variable_type, plot_type):
    col_names = read_json('col_names.json')
    if plot_type == "Boxplot":
        fig = box_plot(series, variable_type,col_names)
        #fig = px.box(series, y=variable_type,notched=True,labels= dict(variable_type=col_names[variable_type]))        
    if plot_type == "Histogram":
        fig = histogram(series, variable_type,col_names)
        #fig = px.histogram(series, x=variable_type)
    if plot_type == "Violin Plot":
        fig = violin(series, variable_type,col_names)
    
    fig.update_layout(width=450, height=275,margin=dict(l=1, r=1, b=1, t=1))
    return (fig)

def streamlit_uni_analysis(series, variable_type, plot_type):
    col_names = read_json('col_names.json')
    sub_col1, sub_col2 = st.beta_columns(2)
    
    if variable_type == 'review_score' or variable_type == "Types of products":
        #st.write("Since Review Scores can only hold values from 1 to 5, this is a categorical variable. Hence it would "
        #         "be more accurate to conduct categorical univariate statistics, as in with a count plot, as opposed "
        #         "to conventional numerical plots like box plot,histogram and violin plot.")
        fig,axes = count_plot(series, variable_type, col_names)
        sub_col1.write(fig)
    else:
        sub_col1.write(uni_plot(series, variable_type, plot_type))  # .show()

    series = series[variable_type]
    describe = pd.DataFrame(series.describe())
    describe.reset_index(inplace=True)
    
    sub_col2.write("")
    sub_col2.write(home.plotly_table (describe,list(describe),width = 450, height = 300))

    
    if (series.infer_objects().dtypes != "object"):
        std = int(series.std())
        max_val = int(series.max())
        if std >= (max_val / 2):
            st.info(
                "This plot has a high standard deviation relative to it's maximum point. This suggests that data in "
                "this plot is not clustered to a mean and is more spread out.")
        else:
            st.info(
                "This plot has a low standard deviation relative to it's maximum. This suggests that most points are "
                "clustered to a mean.")


def univariate(dataframe,variable_type,col2):

    # Opening JSON file
    col_names = read_json('col_names.json')
    col_name_rev = dict((v,k) for k,v in col_names.items())
    
    if variable_type == 'review_score' or variable_type == "Types of products":
        #st.write (variable_type)
        plots = ["Count Plot"]
        #plot_type = st.selectbox("Choose which plot you want", plots)
        plot_type = st.sidebar.radio("Choose which plot you want:", plots)
        streamlit_uni_analysis(dataframe, variable_type, plot_type)
    else:  
        plots = ["Boxplot", "Histogram", "Violin Plot"]
        #plot_type = st.selectbox("Choose which plot you want", plots)
        plot_type = st.sidebar.radio("Choose which plot you want:", plots)
        outlier = st.checkbox('Remove Outliers',value=True)
        if outlier:
            series_wo_outlier = remove_outlier_IQR(dataframe[variable_type])
            streamlit_uni_analysis(pd.DataFrame(series_wo_outlier,columns=[variable_type]), variable_type, plot_type)
        else:
            streamlit_uni_analysis(dataframe, variable_type, plot_type)
            
            
            

def bivariate_tab(dataframe,variable1,variable2):  
    col_names = read_json('col_names.json')
    
    
    col_names = read_json('col_names.json')
    outlier = st.checkbox('Remove Outliers',value=True)
    if (variable1 == 'review_score' or variable1 == "Types of products") or (variable2 == "Types of products" or variable2 == 'review_score'):
        if (variable1 == 'review_score' or variable1 == "Types of products"):
            temp = variable1
            variable1 = variable2
            variable2 = temp
            #temp, variable1, variable2 = variable1,variable2,temp
            
        #outlier = st.checkbox('Remove Outliers')
        #sub_col1, sub_col2 = st.beta_columns(2)
        if outlier:
            dataframe_wo_outlier = remove_outlier_bivariate(dataframe, variable1,None)
            df = dataframe_wo_outlier.copy()
            #st.write(multiple_box_plot(dataframe_wo_outlier,variable1,variable2,5,width=1000, height=400))
        else:
            df = dataframe.copy()
            #st.write("wait for it")
            #n_bins = sub_col2.slider('Bins', 2, 5, 1)
        st.write(multiple_box_plot(dataframe,variable1,variable2,5,width=1000, height=400))
        
    else:
        #outlier = st.checkbox('Remove Outliers')
        sub_col1, sub_col2 = st.beta_columns(2)
        if outlier:
            dataframe_wo_outlier = remove_outlier_bivariate(dataframe, variable1,variable2)
            df = dataframe_wo_outlier.copy()
            #sub_col1.write(scatter_plot(dataframe_wo_outlier,variable1,variable2))
            #n_bins = st.sidebar.slider('Bins', min_value = 1, max_value=5, step=1,value=3)
            #sub_col2.write(multiple_box_plot(dataframe_wo_outlier,variable1,variable2,n_bins,width=500, height=400))
            #st.info("The correlation of", col_names[variable1], "against ", col_names[variable2]," is: ", bivariate_cor(dataframe_wo_outlier,variable1,variable2))
        
        else:
            df = dataframe.copy()
        
        sub_col1.write(scatter_plot(df,variable1,variable2))
        n_bins = st.sidebar.slider('Bins', min_value = 1, max_value=5, step=1,value=3)
        sub_col2.write(multiple_box_plot(df,variable1,variable2,n_bins,width=450, height=400))
        st.info("The correlation of "+col_names[variable1]+" against "+col_names[variable2]+" is: "+str(bivariate_cor(df,variable1,variable2)))

    #print(bin_dataframe(dataframe,variable2,5))
    #print (bin_dataframe(dataframe,variable2,5).dtypes)
    #st.write(type(pd.DataFrame(bin_dataframe(dataframe,variable2,5))))
    #st.dataframe(pd.DataFrame(bin_dataframe(dataframe,variable2,5)))
    #data_frame_new = pd.DataFrame(bin_dataframe(dataframe,variable2,5))
    
    
    
 
def remove_variable (list,variable):
    new_list = []
    for i in list:
        if (i!=variable):
            new_list.append(i)
            
    return(new_list)


def eda(dataframe):
    st.sidebar.markdown("""---""")
    col1,col2 = st.beta_columns(2)
    #st.sidebar.write("Analysis Type:")
    #col2.header("")
    #col1.selectbox("Choose the analysis type: ", analysis_types)
    analysis_types = ['Univariate',
                      'Multivariate',
                     ]
    
    analysis_type  = st.sidebar.selectbox("Analysis Type:", analysis_types)
    
    variable_types = ['price',
                      'freight_value',
                      'review_score',
                      'product_weight_g',
                      'Types of products',
                      'payment_value',
                      'volume',
                      'delivery_days',
                      'estimated_days']
                        
    col_names = read_json('col_names.json')
    col_name_rev = dict((v,k) for k,v in col_names.items())
    

    if analysis_type == "Univariate":
        variable_type_temp = st.sidebar.selectbox("Select Variable:", list(map(col_names.get, variable_types)))
        variable_type = col_name_rev[variable_type_temp]
        univariate(dataframe,variable_type,col2)
    
    if analysis_type == "Multivariate":
        variable_type_temp = st.sidebar.selectbox("Variable 1", list(map(col_names.get, variable_types)))
        variable1 = col_name_rev[variable_type_temp]
        
        #st.write(variable1)
        variable_types2 = remove_variable (variable_types,variable1) #.copy()#.remove(variable1)
        #st.write(variable_types2)
        variable_type_temp2 = st.sidebar.selectbox("Variable 2", list(map(col_names.get, variable_types2)))
        variable2 = col_name_rev[variable_type_temp2]
        
        bivariate_tab(dataframe,variable1,variable2)