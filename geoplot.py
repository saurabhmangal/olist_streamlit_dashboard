# Basic Libraries
import streamlit as st
import numpy as np
import seaborn as sb
from bokeh.models import HoverTool, ColumnDataSource
#from pkg_resources import get_provider

from bokeh.io import output_file, show
from bokeh.tile_providers import get_provider, OSM
from bokeh.plotting import figure, output_file, show
import streamlit.components.v1 as components

sb.set()  # set the default Seaborn style for graphics

# Bokeh maps are in mercator. Convert lat lon fields to mercator units for plotting
def wgs84_to_web_mercator(df, lon, lat):
    k = 6378137
    df["x"] = df[lon] * (k * np.pi / 180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi / 360.0)) * k
    return df


def convert_latlon_tofloat(dataframe):
    # converting all longitude and latitude data to type float, so bokeh can read coordinates
    dataframe['customer_lat'] = dataframe['customer_lat'].astype('float')
    dataframe['customer_lng'] = dataframe['customer_lng'].astype('float')
    dataframe['seller_lat'] = dataframe['seller_lat'].astype('float')
    dataframe['seller_lng'] = dataframe['seller_lng'].astype('float')


def map_prep(dataframe):
    df = wgs84_to_web_mercator(dataframe, 'customer_lng', 'customer_lat')
    convert_latlon_tofloat(dataframe)
    # Establishing a zoom scale for the map. The scale variable will also determine proportions for hexbins and
    # bubble maps so that everything looks visually appealing.
    scale = 2000
    x = df['x']
    y = df['y']
    # The range for the map extents is derived from the lat/lon fields. This way the map is automatically centered on
    # the plot elements.
    x_min = int(x.mean() - (scale * 350))
    x_max = int(x.mean() + (scale * 350))
    y_min = int(y.mean() - (scale * 350))
    y_max = int(y.mean() + (scale * 350))
    return x_min, x_max, y_min, y_max, df

# function takes scale (defined above), the initialized plot object, and the converted dataframe with mercator
# coordinates to create a hexbin map
def hex_map(plot, df, scale, leg_label='Hexbin Heatmap'):
    x = df['x']
    y = df['y']
    r, bins = plot.hexbin(x, y, size=scale * 0.5, hover_color='pink', hover_alpha=0.2, legend_label=leg_label)
    hex_hover = HoverTool(tooltips=[('count', '@c')], mode='mouse', point_policy='follow_mouse', renderers=[r])
    hex_hover.renderers.append(r)
    plot.tools.append(hex_hover)
    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"


def hexmap_plot(dataframe, x_min, x_max, y_min, y_max, df):
    # Defining the map tiles to use. I use OSM, but you can also use ESRI images or google street maps.
    tile_provider = get_provider(OSM)
    scale = 2000
    #x = df['x']
    #y = df['y']
    # Establish the bokeh plot object and add the map tile as an underlay. Hide x and y axis.
    plot = figure(
        #title='Brazil O-list Sales by location',
        plot_width=1000, plot_height=400,
        match_aspect=True,
        tools='wheel_zoom,pan,reset,save',
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        x_axis_type='mercator',
        y_axis_type='mercator'
    )
    plot.grid.visible = True
    map = plot.add_tile(tile_provider)
    map.level = 'underlay'
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    

    # Create the hexbin map
    hex_map(plot=plot,
            df=dataframe,
            scale=scale,
            leg_label='Sale',
            )
    
    show(plot)
            
# create a bubble map.
def bubble_map(plot, df, radius_col, lon, lat, scale, color='orange', leg_label='Bubble Map'):
    radius = []
    for i in df[radius_col]:
        radius.append(i * scale)
    df['radius'] = radius
    source = ColumnDataSource(df)
    c = plot.circle(x='x', y='y', color=color, source=source, size=1, fill_alpha=0.4, radius='radius',
                    legend_label=leg_label, hover_color='red')
    tip_label = '@' + radius_col
    lat_label = '@' + lat
    lon_label = '@' + lon
    circle_hover = HoverTool(
        tooltips=[(radius_col, tip_label), ("Review Score", "@review_score"), ("Weight", "@product_weight_g")],
        mode='mouse', point_policy='follow_mouse', renderers=[c])
    circle_hover.renderers.append(c)
    plot.tools.append(circle_hover)
    # The legend.click_policy method allows us to toggle layer on/off by clicking the corresponding field in the legend. We'll explore this more later!
    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"
    show(plot)

def bubblemap_plot(dataframe, x_min, x_max, y_min, y_max, df):
    # Defining the map tiles to use. I use OSM, but you can also use ESRI images or google street maps.
    tile_provider = get_provider(OSM)
    # Establish the bokeh plot object and add the map tile as an underlay. Hide x and y axis.
    plot = figure(
        #title='Brazil O-list Sales by location',
        plot_width=1000, plot_height=400,
        match_aspect=True,
        tools='wheel_zoom,pan,reset,save',
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        x_axis_type='mercator',
        y_axis_type='mercator'
    )
    plot.grid.visible = True
    map = plot.add_tile(tile_provider)
    map.level = 'underlay'
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    # If in Jupyter, use the output_notebook() method to display the plot in-line. If not, you can use output_file() or another method to save your map.
    #st.write("Check point")
    #output_file('bubblemap.html')

    # Create the bubble map. In this case, circle radius is defined by the amount of fatalities. Any column can be chosen to define the radius.
    bubble_map(plot=plot,
               df=dataframe,
               radius_col='delivery_days',
               leg_label='delivery_days',
               lon='customer_lng',
               lat='customer_lat',
               scale=20)


def hexmap_func(dataframe):
    x_min, x_max, y_min, y_max, df = map_prep(dataframe)
    output_file('hexmap.html')   
    hexmap_plot(dataframe, x_min, x_max, y_min, y_max, df)
    
    # If in Jupyter, use the output_notebook() method to display the plot in-line. If not, you can use output_file()
    # or another method to save your map.

    HtmlFile = open("hexmap.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    #print(source_code)
    components.html(source_code, height=400,width=1000,)


def bubble_func(dataframe):
    x_min, x_max, y_min, y_max, df = map_prep(dataframe)
    bubblemap_plot(dataframe, x_min, x_max, y_min, y_max, df)
    # If in Jupyter, use the output_notebook() method to display the plot in-line. If not, you can use output_file()
    # or another method to save your map.
    output_file('bubblemap.html')
    HtmlFile = open("bubblemap.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    #print(source_code)
    components.html(source_code, width=1000, height=400)


def geo_tab(dataframe):
    st.sidebar.markdown("""---""")
    st.write('## Geospatial Data Analysis')
    col1,col2 = st.beta_columns(2)
    map_types = ['Map - Number of Sales orders',
                 'Map - Time for Delivery']

    map_type = st.sidebar.radio("Map Type:", map_types)

    if map_type == 'Map - Number of Sales orders':
        st.write("### Map - Number of Sales orders ")
        hexmap_func(dataframe)
        st.write("### Location of buyers/sellers ")
        st.write("- We can see the rough distribution of our variables and using geospatial data")
        st.write("- We get an overview into where our sellers and buyers are located for our dataset.")
        st.write("- Most deliveries come from main cities like Sao Paulo and Rio de Janeiro")
        st.write("- There are clusters of deliveries made to various other cities as well (Curitiba, "
            "Joinville, Florianopolis, Londrina, Juiz de Fora, Belo Horizonte).")
        st.write("- Gives indication on where warehouses may be situated, and where we can invest into more delivery services.")
        st.write("- It seems beneficial to invest into delivery trucks and personnel near the main cities, "
            "and perhaps have relatively less resources invested for others. The nature and size of the resources to be "
            "invested can only be decided upon further detailed data analysis.")
        st.write("- When zoomed into specific cities,we see specific neighborhoods attract most customers for the e-commerce business. For "
            "example, within Sao Paulo, we can see the city center hexmap to be a yellowish colour. This "
            "lgihter colour indicates a higher number of sales, showing how this specific area within the city attracts "
            "most customers.")
        st.write("- Lighter colour points show us that most potential and target customers live in that area.")

    if map_type == 'Map - Time for Delivery':
        st.write("### Map - Time for Delivery ")

        bubble_func(dataframe)

        st.write("### Weight ")
        st.write("- Weight of product affects delivery time.")
        st.write("- It takes longer for distributors to deliver a wooden wardrobe as opposed to delivering a book.")
        st.write("- Large radius for the coordinates suggest longer delivery duration.")
        st.write("- Most coordinates with large bubbles, reflect heavy products from interactive legend")
        st.write("- Hence for heavy products in locations like North of Salvador, a longer delivery duration may be "
                 "justified. ")

        st.write("### Review Rating ")
        st.write("- Locations with large bubbles have long delivery durations")
        st.write("- Distributors can work on increasing customer satisfaction")
        st.write("- These locations have relatively low review rating scores (around 1-3). ")
        st.write("- Proves that longer delivery days in these locations is a factor that reduces cutomer satisfaction. ")
        
        st.write(
        "We can use inbuilt functionality within the 'bokeh' module, to derive insights. From this map, "
        "each orange bubble is actually representative of a sale, wherein the radius of the bubble is defined by the "
        "delivery days. For example if a certain order took 11 days to deliver then the bubble would be alot more "
        "bigger and noticeable as compared to a bubble representing 2 day delivery. This helps us gain isight into "
        "the locations that tend to have longer delivery durations, because we can analyse the map to recognise "
        "locations that have larger bubbles as opposed to locations with smaller bubbles. From the map above it is "
        "apparent that locations: 1) West of Floresta da Tijura near Rio de Janeiro 2) Saito 3) Ribeirao Preto 4) "
        "Salvador")
