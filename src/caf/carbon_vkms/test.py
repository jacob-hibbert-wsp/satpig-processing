
import pandas as pd
import numpy as np
import sys


route_data = pd.DataFrame({
    'route_id': [1, 1, 1, 2, 2, 2,3,4,5],
    'origin': ['A', 'A', 'A', 'B', 'B', 'B','A', 'A', 'B'],
    'destination': ['D', 'D', 'D', 'E', 'E', 'E','C','E','C'],
    'speed': [20, 45, 9, 70, 90, 120,60,49,55],
    'distance': [5, 10, 15, 20, 25, 30, 4,45,63],
    #'link': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
})
LINK_DATA_COLUMNS = ["speed", "distance"]
def distance_weighted_mean(data: pd.Series) -> float:
        return np.average(data, weights=route_data.loc[data.index, "distance"])

    # TODO Check there aren't any negative speeds, add warning for any too large speeds > 120kph
    # TODO Write function to calculate speed bands with total distance in each
def speed_bands(data: pd.Series):
    print(data)
    distances, bins = np.histogram(
        data,
        bins=(0, 10, 30, 50, 70,90,110 ,np.inf),
        weights=route_data.loc[data.index, "distance"],
    )
    return pd.Series(distances, index=['0-10', '10-30', '30-50', '50-70', '70-90','90-110','110+'])
    #LOG.info("creating speed bins")

# Apply the functions after grouping by route_id, origin, and destination
grouped = route_data.groupby(['route_id', 'origin', 'destination'])

# Calculate the distance-weighted mean speed for each group
grouped_speed_mean = route_data.groupby(["route_id", "origin", "destination"])[LINK_DATA_COLUMNS].aggregate({"speed": distance_weighted_mean, "distance": "sum"})

# Calculate the speed bands for each group and include the distances
grouped_speed_bands = grouped['speed'].apply(speed_bands).unstack().fillna(0)

# Combine the results into a single DataFrame
route_totals = grouped_speed_mean.join(grouped_speed_bands)
#route_totals=route_totals.droplevel(0, axis=1) 
bins = (0, 10, 30, 50,70,90, 110, np.inf)
bin_names = ['0-10', '10-30', '30-50', '50-70', '70-90','90-110','110+']
route_data["speed_band"] = np.digitize(route_data["speed"], bins)
route_data["speed_band"] = route_data["speed_band"].replace({i: j for i, j in enumerate(bin_names, start=1)})
print(route_data)
route_data = route_data.groupby(["route_id", "origin", "destination", "speed_band"])["distance"].sum().unstack().fillna(0)
print(route_data)
print("\nCombined DataFrame with distance-weighted mean speed and speed bands:")
print(route_totals)
