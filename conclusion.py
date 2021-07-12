import streamlit as st

def rec():
    st.sidebar.markdown("""---""")
    st.title("Conclusion and Recommendation")

    st.write('## Conclusion')
    st.write("From our analysis and calculation, we have created a linear regression model based on a number of factors in order to predict delivery days taken for an order. Although our explained variance value is pretty low, we have managed to increase it a little by reducing outliers and introducing more variables such as distance between sellers and buyers. Using the linear regression model, we calculated the estimated days taken using the model and added it to our dataframe. From there, we checked the correlation between actual days, estimated days and estimated days from our model. As a result we found out that the values from our model actually was closer to the actual delivery days, which was more accurate than what the estimated delivery time given by Olist was. We also calculated the sum of difference between each of them, and the difference between actual days and estimated days was more than double of that compared to the difference between actual days and estimated values from our model. In conclusion, our model has done relatively well given the low R^2 value and it has already come closer than the estimate given by Olist. We also tried new techniques such as Random Forest Regression and Geospatial Analysis using Bokeh Maps.")

    st.write('## Recommendation')
    st.write('### 1) Use machine learning to better gauge estimated delivery duration.')
    st.write("Using our model with a low explained variance, we were able to predict the delivery days taken much better compared to Olist. Thus, Olist should look into improving the data they are expected to give. With better and more relistic data, there would be more transparency between the customer and the e-commerce company allowing for more cutomer satisfaction.")

    st.write('### 2) Increase distribution network.')
    st.write("Distance has highest correlation with delivery days. This suggests that when the buyer and seller are far apart the delivery takes longer. Hence o-list could work onto increasing their distribution network. If O-list were to tap into local distribution networks across the cities of Brazil, and not limit itself to delivering from cities like Rio de Janeiro and Sao Paulo it would do better with delivery time.")

    st.write('### 3) Create more warehouses in certain locations.')
    st.write("From the geospatial analysis it is apparent that there are various pockets in brazil, that have high demand for O-list products, however deliery duration is long, and hence review scores are low. Therefore, O-list could look into creating more warehouses in locations like: West of Floresta da Tijura, Saito, Ribeirao Preto, Salvador. This would allow them to deliver to more consumers, in relatively smaller times.")