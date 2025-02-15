import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from analyse_data import analysis_clustering, analysis_delivery_time_vs_rating
from clean_data import data_clean
from get_data import combine_df, load_website, load_csv
from visualize_results import vistualization


st.set_page_config(layout="wide")

st.title("Welcome To Forecastor")
st.subheader("Forecasting Food Delivery Time")

st.text("Loading the data from the website...")

progress_bar = st.progress(0)

status_text = st.empty()

# df_list = load_website(progress_bar, status_text)

combined_df = load_csv()

progress_bar.progress(100)

# combined_df = combine_df(df_list)

st.text("Data Loaded Successfully")

st.text("Starting to clean the data...")

progress_bar_2 = st.progress(0)

clean_data = data_clean(combined_df, progress_bar_2)

st.text("Data Cleaned Successfully")

st.dataframe(clean_data)

st.text("Starting to visualize the data...")

vistualization(clean_data, st)

analysis_delivery_time_vs_rating(clean_data, st)

analysis_clustering(combined_df, st)
