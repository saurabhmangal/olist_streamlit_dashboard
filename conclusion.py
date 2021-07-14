import streamlit as st

def rec():
    st.sidebar.markdown("""---""")
    st.title("Conclusion and Recommendation")

    st.write('## Conclusion')
    st.write("- We have created a linear regression model based on a number of "
             "factors in order to predict delivery days taken for an order.")
    st.write("- We increased our Explained Variance value by reducing outliers and introducing more variables "
             "like distance between sellers and buyers. Using the linear regression model, we calculated the "
             "estimated days taken using the model and added it to our dataframe.")
    st.write("- We trained our data to follow different Machine Learning models, and analysed which model was most "
             "apt in representing and predicting our data.")
    st.write("- We checked the correlation between actual number of days for delivery, estimated number of days and "
             "estimated days from our model.")
    st.write("- We found that the values from our model actually was closer to the actual delivery days, ")
    st.write("- In conclusion, this tool explore the O-list data set in a interactive setting")
    st.write("- The user is able to explore different variables, and control which variables are used in the analysis")


    st.write('## Recommendation')
    st.write('### 1) Use machine learning to better gauge estimated delivery duration.')
    st.write("- We were able to predict the delivery days taken much better compared to Olist.")
    st.write("- Olist should improve their estimated number os days for delivery.")
    st.write("- More realistic data gives more transparency between O-List and the consumer allowing for more customer "
             "satisfaction.")

    st.write('### 2) Increase distribution network.')
    st.write("- Distance has highest correlation with delivery days.")
    st.write("- When the buyer and seller are far apart the delivery takes longer.")
    st.write("- O-List could increasing their distribution network.")
    st.write("- They could tap into local distribution networks across the cities of Brazil, and not limit itself to "
             "delivering from cities like Rio de Janeiro and Sao Paulo.")

    st.write('### 3) Create more warehouses in certain locations.')
    st.write("- From the geospatial analysis it is apparent that there are various pockets in brazil, that have high "
             "demand for O-list products")
    st.write("- When deliery duration is long, the review scores are low.")
    st.write("- O-list could look into creating more warehouses in locations like: West of Floresta da Tijura, Saito, "
             "Ribeirao Preto, Salvador.")
    st.write("- This would allow more deliveries to more consumers, with shorter number of delivery days.")