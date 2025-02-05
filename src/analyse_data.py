import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def analysis_delivery_time_vs_rating(df):

    print(df.describe())

    df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')

    Average_rating = df['Delivery_person_Ratings'].mean()
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].fillna(round(Average_rating, 1))

    delivery_time = np.array(df['Time_taken(min)'].str.split(" ").str[1]).astype(int)
    ratings = np.array(df['Delivery_person_Ratings'])

    data = {
        'delivery_time': delivery_time,
        'ratings': ratings
    }

    model_input = pd.DataFrame(data)


    mean_input = model_input.groupby(delivery_time)['ratings'].mean()

    mean_input = mean_input.to_frame().reset_index()


    delivery_time = mean_input['index'].values.reshape(-1, 1)  # Reshape to a 2D array
    ratings = mean_input['ratings'].values

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(delivery_time, ratings)

    # Predictions for the model
    ratings_pred = model.predict(delivery_time)

    # Print the coefficients of the linear regression model
    print(f"Intercept (beta_0): {model.intercept_}")
    print(f"Slope (beta_1): {model.coef_[0]}")

    # Model evaluation
    mse = mean_squared_error(ratings, ratings_pred)
    r2 = r2_score(ratings, ratings_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Plot the data and the regression line
    plt.scatter(delivery_time, ratings, color='blue', label="Actual Ratings")
    plt.plot(delivery_time, ratings_pred, color='red', label="Regression Line")
    plt.title('Delivery Time vs. Ratings')
    plt.xlabel('Delivery Time (minutes)')
    plt.ylabel('Average Rating')
    plt.legend()
    plt.text(12.0,4.5, 'Insight:\nWe are able to get a linear regression, \nbut the data is not \nfollowing that regression closely.',
             fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.savefig('avg_rating_by_time_linear_regression.png', dpi=300, bbox_inches='tight')
    plt.show()

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(delivery_time.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly, ratings)

    y_pred = model.predict(X_poly)

    plt.scatter(delivery_time, ratings, color='blue', label='Actual Ratings')
    plt.plot(delivery_time, y_pred, color='red', label='Polynomial Fit (Degree 2)')
    plt.title('Delivery Time vs. Ratings (Polynomial Regression)')
    plt.xlabel('Delivery Time (minutes)')
    plt.ylabel('Average Rating')
    plt.legend()
    plt.text(10,4.5, 'Insight: \nWe are able to get a polynomial regression. \nHere, the graph is closer to the data.',
             fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.savefig('avg_rating_by_time_polynomial_regression.png', dpi=300, bbox_inches='tight')
    plt.show()

    X = X_poly
    y = ratings

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    degree = 2
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R-squared: {r2:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")


def analysis_clustering(df):
    # df = df1.merge(df2, on='ID', how='outer').merge(df3, on='ID', how='outer')
    # df = df[1:]
    # df.head(2)

    df['Vehicle_condition'] = df['Vehicle_condition'].astype('int64')
    df['Restaurant_latitude'] = df['Restaurant_latitude'].astype('float64')
    df['Restaurant_longitude'] = df['Restaurant_longitude'].astype('float64')
    df['Delivery_location_latitude'] = df['Delivery_location_latitude'].astype('float64')
    df['Delivery_location_longitude'] = df['Delivery_location_longitude'].astype('float64')

    print(df.info())
    df['Delivery_location_latitude'], df['Delivery_location_longitude']

    df['Restaurant_latitude'] = pd.to_numeric(df['Restaurant_latitude']).abs()
    df['Restaurant_longitude'] = pd.to_numeric(df['Restaurant_longitude']).abs()
    df['Delivery_location_latitude'] = pd.to_numeric(df['Delivery_location_latitude']).abs()
    df['Delivery_location_longitude'] = pd.to_numeric(df['Delivery_location_longitude']).abs()

    df = df[~((df['Restaurant_latitude'] <= 1) & (df['Restaurant_longitude'] <= 1) &(df['Delivery_location_latitude'] <= 1) &(df['Delivery_location_longitude'] <= 1))]
    plt.figure(figsize=(8, 6))
    # plt.scatter(df['Restaurant_latitude'], df['Restaurant_longitude'], color='red', marker='o')
    plt.scatter(df['Delivery_location_latitude'], df['Delivery_location_longitude'], color='blue')

    # Adding labels
    plt.title('Latitude and Longitude Plot')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Display the plot
    plt.grid(True)
    plt.text(10,88, 'Insight: PLotting latitude vs longitude.',
             fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.savefig('lat_long_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    X = df
    # y = df['Restaurant_longitude']

    data = pd.DataFrame(X, columns=['Delivery_location_latitude', 'Delivery_location_longitude'])

    # Visualize the data
    plt.scatter(data['Delivery_location_latitude'], data['Delivery_location_longitude'], s=10)
    plt.title('Data Visualization')
    plt.xlabel('Delivery_location_latitude')
    plt.ylabel('Delivery_location_longitude')
    plt.text(10,88, 'Insight: PLotting latitude vs longitude.',
             fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.savefig('delivery_location_lat_long.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Determine the optimal number of clusters using the Elbow Method
    inertia = []
    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    # Plotting the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_range)
    plt.grid()
    plt.text(1,4.5, 'Insight:\nWe can see that the inflexion point\n in the elbow graph is at 4.',
             fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.savefig('elbow_graph.png', dpi=300, bbox_inches='tight')
    plt.show()

    k = 4  # Optimal k from the elbow method
    kmeans = KMeans(n_clusters=k)
    data['Cluster'] = kmeans.fit_predict(data)

    # Print the cluster centers
    print("Cluster Centers:\n", kmeans.cluster_centers_)

    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Delivery_location_latitude'], data['Delivery_location_longitude'], c=data['Cluster'], s=10, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=100, alpha=0.75, label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel('Delivery_location_latitude')
    plt.ylabel('Delivery_location_longitude')
    plt.legend()
    plt.text(10,88, 'Insight:\nWith the clusting value as 4,\nwe are able to get well defined clusters using \nthe latitude and the longitude.',
             fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.savefig('clustering.png', dpi=300, bbox_inches='tight')
    plt.show()