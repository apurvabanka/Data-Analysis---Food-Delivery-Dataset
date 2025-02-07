import numpy as np
import pandas as pd


def data_clean(df, progress_bar):

    print(df.info())

    print("################################################")
    print("Removing NULL Values")
    print("################################################")


    null_value_removed = removing_null_values(df)

    progress_bar.progress(30)

    print("################################################")
    print("Cleaning Latitude and Longitude")
    print("################################################")


    lat_long_clean = cleaning_lat_long(null_value_removed)

    progress_bar.progress(60)

    print("################################################")
    print("Cleaning Data Types")
    print("################################################")


    datatype_clean = cleaning_datatypes(lat_long_clean)

    progress_bar.progress(100)

    return datatype_clean

def removing_null_values(df):

    df['Vehicle_condition'] = df['Vehicle_condition'].astype('int64')
    df['Restaurant_latitude'] = df['Restaurant_latitude'].astype('float64')
    df['Restaurant_longitude'] = df['Restaurant_longitude'].astype('float64')
    df['Delivery_location_latitude'] = df['Delivery_location_latitude'].astype('float64')
    df['Delivery_location_longitude'] = df['Delivery_location_longitude'].astype('float64')

    # Replacing 'NaN' with NaN.
    df = df.replace('NaN', np.nan, regex=True)
    print("Finding SUM of Nan values")
    # Finding the count of NaN for each column
    print(df.isna().sum())
    print("------------------------")
    print("Finding SUM of 0 values")
    # Finding columns with count as 0.
    print((df == 0).sum())

    return df

def cleaning_lat_long(df):

    threshold = 0.01

    # Task 1: Identify the valid range for Restaurant Latitude and Longitude
    valid_restaurant_latitude = df[df['Restaurant_latitude'] > threshold]['Restaurant_latitude']
    valid_restaurant_longitude = df[df['Restaurant_longitude'] > threshold]['Restaurant_longitude']

    # Task 2: Identify the valid range for Delivery Latitude and Longitude
    valid_delivery_latitude = df[df['Delivery_location_latitude'] > threshold]['Delivery_location_latitude']
    valid_delivery_longitude = df[df['Delivery_location_longitude'] > threshold]['Delivery_location_longitude']

    # Task 3: Calculate the min and max for each of the valid columns
    min_rest_lat, max_rest_lat = valid_restaurant_latitude.min(), valid_restaurant_latitude.max()
    min_rest_long, max_rest_long = valid_restaurant_longitude.min(), valid_restaurant_longitude.max()
    min_del_lat, max_del_lat = valid_delivery_latitude.min(), valid_delivery_latitude.max()
    min_del_long, max_del_long = valid_delivery_longitude.min(), valid_delivery_longitude.max()

    # Task 4: Replace zero or near-zero values with random values from the respective range
    df['Restaurant_latitude'] = df['Restaurant_latitude'].apply(
        lambda x: round(np.random.uniform(min_rest_lat, max_rest_lat), 4) if x <= threshold else x
    )
    df['Restaurant_longitude'] = df['Restaurant_longitude'].apply(
        lambda x: round(np.random.uniform(min_rest_long, max_rest_long), 4) if x <= threshold else x
    )
    df['Delivery_location_latitude'] = df['Delivery_location_latitude'].apply(
        lambda x: round(np.random.uniform(min_del_lat, max_del_lat), 4) if x <= threshold else x
    )
    df['Delivery_location_longitude'] = df['Delivery_location_longitude'].apply(
        lambda x: round(np.random.uniform(min_del_long, max_del_long), 4) if x <= threshold else x
    )

    # Display the updated DataFrame
    print(df[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']])

    return df

def cleaning_datatypes(df):

    df['Time_Ordered_new'] = pd.to_datetime(df['Time_Orderd'], errors='coerce')
    df['Time_Order_picked_new'] = pd.to_datetime(df['Time_Order_picked'], errors='coerce')

    # Step 1: Calculate the time difference where "Time Ordered" is not NULL
    df['Time_Difference'] = df['Time_Order_picked_new'] - df['Time_Ordered_new']

    # Step 2: Calculate the average time difference (exclude rows where Time Ordered is NULL)
    average_time_diff = df['Time_Difference'].mean()

    # Step 3: Replace NULL values in "Time Ordered" by subtracting the average time difference from "Time Packed"
    df['Time_Ordered_new'] = df.apply(
        lambda row: row['Time_Order_picked'] - average_time_diff if pd.isnull(row['Time_Ordered_new']) else row['Time_Ordered_new'],
        axis=1
    )

    # Drop the Time_Difference column if it's no longer needed
    df.drop('Time_Difference', axis=1, inplace=True)

    # Display the updated DataFrame
    print(df[['Time_Ordered_new', 'Time_Order_picked_new']])

    #Convert 'Delivery_person_Age' to numeric
    df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')

    #Replace null values in 'Delivery_person_Age' with the average
    Average_age = df['Delivery_person_Age'].mean()
    df['Delivery_person_Age'] = df['Delivery_person_Age'].fillna(int(Average_age))

    #Convert 'Delivery_person_Ratings' to numeric
    df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')

    #Replace null values in 'Delivery_person_Ratings' with average(1 decimal point)
    Average_rating = df['Delivery_person_Ratings'].mean()
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].fillna(round(Average_rating, 1))

    #Convert 'Delivery_person_Ratings' to numeric
    df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')

    #Replace null values with the most frequent value which is the mode of the data
    most_frequent_value = df['multiple_deliveries'].mode()[0]
    df['multiple_deliveries'] = df['multiple_deliveries'].fillna(most_frequent_value)

    return df
