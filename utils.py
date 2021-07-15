import plotly.graph_objects as go
import seaborn as sb
import json
import pandas as pd
import streamlit as st
import plotly.express as px


def plotly_table (dataframe,col_names,width = 1100, height = 400):
    fig = go.Figure(data=[go.Table(
    header=dict(values=col_names,
                fill_color='paleturquoise',
                #cells=dict( height=100),
                align='center'),
    cells=dict(values=dataframe.round(decimals=2).transpose().values.tolist(),
               fill_color='lavender',
               alignsrc="center",
               #cells=dict( height=50),
               align='center'))
    ])
    fig.update_layout(width=width, height=height,margin=dict(l=1, r=1, b=1, t=1))
    #fig.show()
    
    return(fig)
    
    
def read_json(file_name = 'col_names.json'):
    # Opening JSON file
    with open(file_name) as json_file:
        col_names = json.load(json_file)
    return col_names

def box_plot(series, variable_type, col_names):
    fig = go.Figure()
    # Use x instead of y argument for horizontal plot
    fig = px.box(series, y=variable_type)
    #fig.add(go.Box(x=series[variable_type]))
    fig.update_layout(
    yaxis_title=col_names[variable_type],
    )
    fig.layout.plot_bgcolor = 'lavender'
    #fig.layout.showline=True
    return(fig)
    
def histogram(series, variable_type, col_names):
    fig = go.Figure()
    # Use x instead of y argument for horizontal plot
    fig = px.histogram(series, x=variable_type)
    #fig.add(go.Box(x=series[variable_type]))
    fig.update_layout(
    xaxis_title=col_names[variable_type],
    )
    fig.layout.plot_bgcolor = 'lavender'
    #fig.layout.showline=True
    return(fig)
    
def violin(series, variable_type, col_names):
    fig = go.Figure()
    # Use x instead of y argument for horizontal plot
    fig = px.violin(series, y=variable_type)
    #fig.add(go.Box(x=series[variable_type]))
    fig.update_layout(
    yaxis_title=col_names[variable_type],
    )
    fig.layout.plot_bgcolor = 'lavender'
    #fig.layout.showline=True
    return(fig)

def count_plot(series, variable_type, col_names):
    sb.set(rc={'axes.facecolor':'lavender', 'figure.facecolor':'white'})
    sb.set(font_scale = 1)
    fig, axes = plt.subplots(figsize=(5, 3))
    sb.countplot(x = variable_type, data = series,saturation=0.75)
    axes.set_xlabel(col_names[variable_type], fontsize = 10)
    axes.set_ylabel("Count", fontsize = 10)
    #axes.set_yticklabels(axes.get_yticks(), size = 8)
    _, xlabels = plt.xticks()
    axes.set_xticklabels(xlabels, size=8)
    _, ylabels = plt.yticks()
    axes.set_yticklabels(ylabels, size=8)

    fig.tight_layout() 
    return(fig,axes)

@st.cache(allow_output_mutation=True)
def remove_outlier_IQR(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    series_wo_outlier = series[~((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR)))]
    #print(series_wo_outlier)
    return series_wo_outlier
    
def remove_outlier_bivariate(dataframe, variable1, variable2):
    if variable2 != None:
        variable_list = [variable1,variable2]
    else:
        variable_list = [variable1]
    for i in variable_list:
        dataframe[i] = remove_outlier_IQR(dataframe[i])
        dataframe = dataframe.dropna().copy()
    return (dataframe)
    
def remove_outlier_dataframe(dataframe, col_list):
    for i in col_list:
        dataframe[i] = remove_outlier_IQR(dataframe[i])
        dataframe = dataframe.dropna().copy()
    return (dataframe)
    
    
@st.cache    
def remove_list_parameters(list_from_remove,list_to_remove):
    new_list = []
    for i in list_from_remove:
        if i in list_to_remove:
            pass
        else:
            new_list.append(i)    
    return(new_list)
    
# function to create multivariate heatmap
def heatmap(numDF):
    fig = plt.figure(figsize=(6, 6))
    sb.heatmap(numDF.corr(), vmin=-1, vmax=1, annot=True, fmt=".2f")
    return fig


# function to compute analysis based on summary statistics
@st.cache
def bivariate_cor(dataframe, variable1, variable2):
    cor = dataframe[variable1].corr(dataframe[variable2])
    cor = round(cor, 2)
    return cor

# function to plot a plotly scatter plot
@st.cache
def scatter_plot(dataframe, variable1,variable2):
    col_names = read_json('col_names.json')
    fig = px.scatter(dataframe, y=variable1, x=variable2)
    fig.update_layout(width=500, height=400,margin=dict(l=1, r=1, b=1, t=1))
    fig.update_layout(
    yaxis_title=col_names[variable1],
    xaxis_title=col_names[variable2],
    )
    fig.layout.plot_bgcolor = 'lavender'
    return fig

def bin_dataframe(df,variable,n_bins):
    dataframe = df.copy()
    dataframe["new_col"] = pd.cut(dataframe[variable],bins = n_bins)
    dataframe = dataframe.drop(columns=[variable]).copy()
    #print (dataframe)
    dataframe[variable] = dataframe["new_col"].astype("str")
    dataframe = dataframe.drop(columns=["new_col"]).copy()
    #print (dataframe)
    #print (dataframe.infer_objects().dtypes)
    return(dataframe)

def multiple_box_plot(dataframe,variable1,variable2,n_bins,width=500, height=400):  
    col_names = read_json('col_names.json')
    if (variable1 == 'review_score' or variable1 == "Types of products") or (variable2 == "Types of products" or variable2 == 'review_score'):
        fig = px.box(dataframe, y=variable1, x=variable2)
    else:
        fig = px.box(bin_dataframe(dataframe,variable2,n_bins), y=variable1, x=variable2)
    fig.update_layout(width=width, height=height,margin=dict(l=1, r=1, b=1, t=1))
    fig.update_layout(
    yaxis_title=col_names[variable1],
    xaxis_title=col_names[variable2],
    )
    fig.layout.plot_bgcolor = 'lavender'
    return(fig)
    

    

