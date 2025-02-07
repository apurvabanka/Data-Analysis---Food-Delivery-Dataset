import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import extract_time_taken

def vistualization(df, st):

    plot = time_taken_vs_rating_visualization(df, st)

    day_part_visualization(df, st)

    distribution_of_delivery_visualisation(df, st)

    vehicle_type_visualisation(df, st)

    return plot


def time_taken_vs_rating_visualization(df, st):
    time_taken = df['Time_taken(min)'].str.split(" ").str[1]

    time_taken = pd.DataFrame(time_taken)

    time_taken["Time_taken(min)"] = time_taken["Time_taken(min)"].astype(int)

    print(time_taken.max())
    print(time_taken.min())

    bins = [10, 20, 30, 40, 50 , 60]

    time_taken['interval'] = pd.cut(time_taken["Time_taken(min)"], bins)

    bin_count = time_taken['interval'].value_counts().sort_index()

    time_taken['rating'] = df['Delivery_person_Ratings'].astype(float)

    average_rating_per_interval = time_taken.groupby('interval')['rating'].mean() - 4

    plt.figure(figsize=(12, 6))

    # Plot the bin count with a legend
    plt.subplot(1, 2, 1)
    bin_count.plot(kind='bar')
    plt.title("Delivery Counts per Time Interval", fontsize=12)
    plt.xlabel("Time Intervals (min)", fontsize=12)
    plt.ylabel("Frequency")
    plt.legend(["Frequency"], loc='upper right')
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot the average ratings per interval with a legend
    plt.subplot(1, 2, 2)

    average_rating_per_interval.plot(kind='bar', color='orange')
    plt.title("Average Ratings per Time Interval (Offset by -4)")
    plt.xlabel("Time Intervals (min)", fontsize=12)
    plt.ylabel("Average Ratings", fontsize=12)
    plt.legend(["Average Rating"], loc='upper right')
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plots
    plt.figtext(0.5, 0.02, "As we can see on the left, the number of orders decreases as time increases. "
                "On the right, ratings also tend to decrease with longer delivery times.", 
                ha="center", fontsize=10, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    plt.tight_layout(rect=[0, 0.05, 1, 1])  
    plt.savefig('time_taken_vs_rating_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    st.pyplot(plt)

    return plt


def day_part_visualization(df, st):
    df["order_time"] = df["Order_Date"].astype(str) + " " + df["Time_Orderd"].astype(str)
    df["order_pickup_time"] = df["Order_Date"].astype(str) + " " + df["Time_Order_picked"].astype(str)

    df["order_time"] = pd.to_datetime(df["order_time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    df["order_pickup_time"] = pd.to_datetime(df["order_pickup_time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")

    mean_order_time = df["order_time"].mean()
    mean_order_pickup_time = df["order_pickup_time"].mean()
    df["order_time"] = df["order_time"].fillna(mean_order_time)
    df["order_pickup_time"] = df["order_pickup_time"].fillna(mean_order_pickup_time)

    df["time_taken_min"] = df["Time_taken(min)"].apply(lambda x: extract_time_taken(x) if isinstance(x, str) else [])

    df_exploded = df.explode('time_taken_min')
    mean_time_taken = df_exploded['time_taken_min'].mean()

    df['Order_hour'] = df['order_time'].dt.hour

    def categorize_time_of_day(hour):
        if 5 <= hour < 11:
            return 'Morning'
        elif 11 <= hour < 16:
            return 'Afternoon'
        elif 16 <= hour < 20:
            return 'Evening'
        else:
            return 'Night'

    df['Part_of_Day'] = df['Order_hour'].apply(categorize_time_of_day)

    df['time_taken_min'] = df['time_taken_min'].apply(lambda x: x[0] if isinstance(x, list) else x)

    avg_delivery_time_by_part_of_day = df.groupby('Part_of_Day')['time_taken_min'].mean().reset_index()
    print("Average Delivery Time by Part of Day:")
    print(avg_delivery_time_by_part_of_day)

    df["order_time"] = df["Order_Date"].astype(str) + " " + df["Time_Orderd"].astype(str)
    df["order_pickup_time"] = df["Order_Date"] + " " + df["Time_Order_picked"]

    # Step 2: Converting the columns to datetime
    df["order_time"] = pd.to_datetime(df["order_time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    df["order_pickup_time"] = pd.to_datetime(df["order_pickup_time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")

    mean_order_time = df["order_time"].mean()
    mean_order_pickup_time = df["order_pickup_time"].mean()
    df["order_time"] = df["order_time"].fillna(mean_order_time)
    df["order_pickup_time"] = df["order_pickup_time"].fillna(mean_order_pickup_time)


    df["time_taken_min"] = df["Time_taken(min)"].apply(lambda x: extract_time_taken(x) if isinstance(x, str) else [])

    df_exploded = df.explode('time_taken_min')
    mean_time_taken = df_exploded['time_taken_min'].mean()

    df['Order_hour'] = df['order_time'].dt.hour

    def categorize_time_of_day(hour):
        if 5 <= hour < 11:
            return 'Morning'
        elif 11 <= hour < 16:
            return 'Afternoon'
        elif 16 <= hour < 20:
            return 'Evening'
        else:
            return 'Night'

    df['Part_of_Day'] = df['Order_hour'].apply(categorize_time_of_day)


    # print(df[['Order_hour', 'Part_of_Day']].head())
    # Extract the first element from each list in the `time_taken_min` column
    df['time_taken_min'] = df['time_taken_min'].apply(lambda x: x[0] if isinstance(x, list) else x)
    # df['time_taken_min']

    avg_delivery_time_by_part_of_day = df.groupby('Part_of_Day')['time_taken_min'].mean().reset_index()
    print("Average Delivery Time by Part of Day:")
    print(avg_delivery_time_by_part_of_day)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_delivery_time_by_part_of_day, x='Part_of_Day', y='time_taken_min')
    plt.title('Average Delivery Time by Part of Day')
    plt.xlabel('Part of Day')
    plt.ylabel('Average Delivery Time (minutes)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for index, value in enumerate(avg_delivery_time_by_part_of_day['time_taken_min']):
        plt.text(index, value + 0.5, f'{value:.1f}', ha='center', fontsize=10)

    plt.text(-0.5, max(avg_delivery_time_by_part_of_day['time_taken_min']) + 2, 
            "Average delivery time is higher in the evening and night.\n"
            "It is lower in the morning compared to other parts of the day.\n"
            "The overall average delivery time is around 23 minutes.\n", 
            fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig('day_part_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    st.pyplot(plt)
    


    df['Order_day'] = df['order_time'].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    df['Order_day'] = pd.Categorical(df['Order_day'], categories=day_order, ordered=True)

    heatmap_data_part_of_day = df.pivot_table(
        values='time_taken_min', 
        index='Order_day', 
        columns='Part_of_Day', 
        aggfunc='mean'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data_part_of_day, annot=True, fmt=".1f", cmap='coolwarm')
    plt.title('Heatmap of Average Delivery Time with Part of Day, Day of Week')
    plt.xlabel('Part of Day')
    plt.ylabel('Day of Week')
    plt.text(0, 2, 'High on Wed evening', fontsize=12, color='red', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='red'))
    plt.text(2.9, 7.6, 'Low in the mornings', fontsize=12, color='green', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='green'))

    plt.savefig('heat_map_avg_delivery_time.png', dpi=300, bbox_inches='tight')
    plt.show()

    st.pyplot(plt)


    avg_delivery_time_by_hour = df.groupby('Order_hour')['time_taken_min'].mean().reset_index()
    min_time = avg_delivery_time_by_hour['time_taken_min'].min()
    max_time = avg_delivery_time_by_hour['time_taken_min'].max()
    min_hour = avg_delivery_time_by_hour[avg_delivery_time_by_hour['time_taken_min'] == min_time]['Order_hour'].iloc[0]
    max_hour = avg_delivery_time_by_hour[avg_delivery_time_by_hour['time_taken_min'] == max_time]['Order_hour'].iloc[0]

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=avg_delivery_time_by_hour, x='Order_hour', y='time_taken_min', marker='o')
    plt.title('Average Delivery Time by Hour of the Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Delivery Time (minutes)')
    plt.xticks(range(24))
    plt.grid()
    plt.annotate(f'Lowest: {min_time:.2f} min', xy=(min_hour, min_time),
             xytext=(min_hour + 1, min_time + 2),
             arrowprops=dict(facecolor='green', arrowstyle='->'),
             fontsize=10, color='green')
    plt.annotate(f'Highest: {max_time:.2f} min', xy=(max_hour, max_time),
             xytext=(max_hour - 3, max_time - 2),
             arrowprops=dict(facecolor='red', arrowstyle='->'),
             fontsize=10, color='red')
    
    text_x = 0.5  # Adjust as needed for text placement
    text_y = min_time + (max_time - min_time) * 0.7
    description = (
        "The graph shows the average delivery time by hour of the day. Starting from 12AM, "
        "the delivery time decreases until 9AM, then gradually increases till 2PM. "
        "It decreases slightly before rising again till 9PM, after which it decreases again. "
        "This pattern repeats daily. The hour with the least delivery time is 8AM-9AM, "
        "while the highest is 8PM-9PM."
    )
    plt.text(text_x, text_y, description, fontsize=10, wrap=True, color='black', va='top', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# Display the plot
    plt.savefig('avg_delivery_time_by_hour.png', dpi=300, bbox_inches='tight')
    plt.show()

    st.pyplot(plt)

def delivery_time_distribution(delivery_time, st):
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(delivery_time, bins=20, color='skyblue', edgecolor='black')
    plt.title('Delivery Time Distribution')
    plt.xlabel('Delivery Time (minutes)')
    plt.ylabel('Frequency')

    plt.text(30, max(plt.hist(delivery_time, bins=20)[0]) * 0.8,
             "Most deliveries occur between 10–35 minutes.\nPeak frequency: 20–25 minutes.",
             fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))


    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=delivery_time, color='lightblue')
    plt.title('Delivery Time Boxplot')
    plt.xlabel('Delivery Time (minutes)')
    plt.text(delivery_time.median() - 5, 0.4,
             f"Median: {delivery_time.median():.1f} minutes\nWhiskers extend beyond the IQR.",
             fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.savefig('delivery_time_box_plot.png', dpi=300, bbox_inches='tight')
    
    st.pyplot(plt)


def distribution_of_delivery_visualisation(df, st):
    df['time_taken_min'] = df["Time_taken(min)"].apply(lambda x: extract_time_taken(x) if isinstance(x, str) else [])
    df_exploded = df.explode('time_taken_min')

    delivery_time_distribution(df_exploded["time_taken_min"], st)

def vehicle_type_visualisation(df, st):
    df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')

    # Drop rows with NaN values in 'multiple_deliveries' column
    df = df.dropna(subset=['multiple_deliveries'])

    # Group by 'Type_of_vehicle' and calculate the average multiple deliveries
    grouped_data = df.groupby('Type_of_vehicle')['multiple_deliveries'].mean().reset_index()

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(grouped_data['multiple_deliveries'],
                                       labels=grouped_data['Type_of_vehicle'],
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=plt.cm.tab10.colors
)

    plt.title('Proportion of Average Multiple Deliveries by Vehicle Type')
    plt.text(0.0, -1.2, 
         "Insight:\nMotorcycles dominate the proportion,\nhandling more multiple deliveries compared to\n"
         "electric scooters, bicycles, and scooters.",
         fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('avg_multiple_delivery_by_vehicle.png', dpi=300, bbox_inches='tight')
    plt.show()

    st.pyplot(plt)

    # Ensure the required columns are present and filter out rows with missing values
    if 'City' in df.columns and 'Delivery_person_Ratings' in df.columns:
        # Convert 'Delivery_person_Ratings' to numeric and drop rows with NaNs
        df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
        df = df.dropna(subset=['City', 'Delivery_person_Ratings'])
        df = df[~df['City'].str.strip().isin(['NaN', ''])]  # Remove rows where City is 'NaN' or empty
        
        # Plotting Histogram: Frequency Distribution of Ratings by Location Type
        location_types = df['City'].unique()
        plt.figure(figsize=(12, 8))
        
        for i, location_type in enumerate(location_types, 1):
            plt.subplot(len(location_types), 1, i)
            subset = df[df['City'] == location_type]
            plt.hist(subset['Delivery_person_Ratings'], bins=5, edgecolor='black', color='skyblue')
            plt.title(f'Histogram of Customer Ratings - {location_type}')
            plt.xlabel('Customer Rating')
            plt.ylabel('Frequency')

        plt.gcf().text(0.35, 0.45,
                   ("Insight:\n"
                    "Both Urban and Metropolitan areas have a high concentration of ratings around 4 to 5, \n"
                    "with very few low ratings. In contrast, the Semi-Urban area shows a more even spread \n"
                    "across ratings, with notable frequencies from 4.2 to 5. This suggests a slightly higher \n"
                    "and more varied satisfaction level in these areas.\n"),
                   fontsize=8, ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        plt.suptitle('Histogram of Customer Ratings by Location Type', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the suptitle and insight text
        plt.savefig('customer_rating_by_location.png', dpi=300, bbox_inches='tight')
        plt.show()

        st.pyplot(plt)

    else:
        print("The required columns 'City' and 'Delivery_person_Ratings' are not present in the dataset.")

