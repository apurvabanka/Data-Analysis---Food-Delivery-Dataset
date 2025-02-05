from analyse_data import analysis_clustering, analysis_delivery_time_vs_rating
from clean_data import data_clean
from get_data import combine_df, load_website
import pandas as pd

from visualize_results import vistualization


if __name__ == "__main__":
    df_list = load_website()

    combined_df = combine_df(df_list)

    clean_data = data_clean(combined_df)

    print(clean_data)

    vistualization(clean_data)

    analysis_delivery_time_vs_rating(clean_data)

    analysis_clustering(combined_df)


