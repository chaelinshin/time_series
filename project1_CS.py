# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import functions as func

# bring data
df = pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv')
df.info()

df['date_time'] = pd.to_datetime(df['date_time'])
df['date'] = pd.to_datetime(df['date_time'])
traffic = df['traffic_volume'] # dependent variable

# holiday, weather_main, and weather_description are categorical variables
print(df['holiday'].unique()) # 11 holidays + None (12 in total)
print(df['weather_main'].unique()) # 11 categories
print(df['weather_description'].unique()) # 38 categories

# encoding categorical data
# holiday --> 1 if holiday (true); 0 if not holiday (false)
# create new column, 'holiday_tf' that contains either 0 or 1
df['holiday_tf'] = np.where(df['holiday'].str.contains('None'), 0, 1)
print(df['holiday_tf'].unique()) # there is only 0 or 1
df = df.drop('holiday', 1) # drop 'holiday' column because we don't need it anymore

# label encoding for weather_main and weather_description
lb_make = LabelEncoder()
df['weather_main_lb'] = lb_make.fit_transform(df['weather_main'])
df['weather_description_lb'] = lb_make.fit_transform(df['weather_description'])

# rename columns
df = df.rename(columns={'rain_1h': 'rain', 'snow_1h': 'snow',
                        'clouds_all': 'clouds', 'weather_main': 'weather_short',
                        'weather_description': 'weather_long',
                        'holiday_tf': 'holiday', 'weather_main_lb': 'weather_short2',
                        'weather_description_lb': 'weather_long2'})

df = df.set_index('date_time')
traffic = df['traffic_volume']

# plot of traffic_volume vs. time
plt.figure()
plt.plot(traffic, label='Traffic Volume', color='green')
plt.xlabel("Time")
plt.ylabel("Volume")
plt.title("Traffic Volume Over Time")
plt.show()

df = df[15982:]
traffic = df['traffic_volume']

# plot of traffic_volume vs. time
plt.figure()
plt.plot(traffic, label='Traffic Volume', color='green')
plt.xlabel("Time")
plt.ylabel("Volume")
plt.xticks(rotation=12)
plt.title("Traffic Volume Over Time")
plt.show()

# correlation matrix (seaborn heatmap)
corr = df.corr()

ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(120,170,n=50), square=True)
bottom, top = ax.get_ylim()
ax.set_title('Correlation Matrix of Traffic Dataset')
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=17, horizontalalignment='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
plt.show()

# correlation coefficient
# traffic_volume and temp
temp = df['temp']
r_temp = func.correlation_coefficient_cal(traffic, temp)
print(f"Correlation coefficient between traffic and temp: {r_temp:.2f}")

# traffic_volume and rain
rain = df['rain']
r_rain = func.correlation_coefficient_cal(traffic, rain)
print(f"Correlation coefficient between traffic and rain: {r_rain:.2f}")

# traffic_volume and snow
snow = df['snow']
r_snow = func.correlation_coefficient_cal(traffic, snow)
print(f"Correlation coefficient between traffic and snow: {r_snow:.2f}")

# traffic_volume and clouds
cloud = df['clouds']
r_cloud = func.correlation_coefficient_cal(traffic, cloud)
print(f"Correlation coefficient between traffic and cloud: {r_cloud:.2f}")

# preprocessing
# duplicates
dup_dict = {"holiday": "first", "temp": "mean", "rain": "mean", "snow": "mean",
            "clouds": "mean", "weather_short": "first", "weather_long": "first",
            "traffic_volume": "mean", "weather_short2": "first",
            "weather_long": "first"}

# selects all duplicated observations
duplicates = df[df["date"].duplicated(keep=False)]
duplicates_ = duplicates.groupby("date").agg(dup_dict).reset_index()
non_duplicates = df[~df["date"].duplicated(keep=False)]
df_ = non_duplicates.append(duplicates_).sort_values("date")

# fill gaps
df_final = df_.set_index("date").resample("H").first().bfill()

# save cleaned file
df_final.to_csv(r'Project/data/Metro_Traffic_clean.csv')


