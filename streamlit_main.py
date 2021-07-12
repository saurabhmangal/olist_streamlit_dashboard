# Basic Libraries
import home
import geoplot
import conclusion
import eda
import machine_learning
import streamlit as st
import pandas as pd
#import neural_net

st.set_page_config(page_title=None, page_icon=None, layout='wide', initial_sidebar_state='expanded')
# Code for grey sidebar
#st.markdown(
#"""
#<style>
#.css-1aumxhk {
#background-color: #404040;
#background-image: none;
#color: #ffffff;
#}
#</style>
#""",
#unsafe_allow_html=True,)

# creating individual dataframes for all numerical variables
dataframe = pd.read_csv("ExportDataFrame.csv", header=0)


# Sidebar Navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:',
    ['Home',
     'Explore Data',
     'Geospatial Plots & Analysis',
     #'Machine Learning',
     'Machine Learning',
     'Conclusion and Recommendation','neural_net'])

if options == 'Home':
    home.home(dataframe)
elif options == 'Explore Data':
    eda.eda(dataframe)
elif options == 'Geospatial Plots & Analysis':
    geoplot.geo_tab(dataframe)
#elif options == 'Machine Learning':
#    MLtab.ML_tab(dataframe)
elif options == 'Conclusion and Recommendation':
    conclusion.rec()
elif options == 'Machine Learning':
    machine_learning.ml_tab(dataframe)
#elif options == 'neural_net':
#    neural_net.neural_net(dataframe)
