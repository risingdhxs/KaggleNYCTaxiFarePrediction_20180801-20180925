

%matplotlib inline

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
pd.options.display.max_rows=10

def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    df['abs_dist_travel'] = df.abs_diff_longitude + df.abs_diff_latitude

def add_loc_bias_feature(df):
    lat_mean=40.75
    long_mean=-73.97
    df['up_diff_center'] = np.sqrt((df.pickup_latitude-lat_mean)**2 + (df.pickup_longitude-long_mean)**2)
    df['off_diff_center'] = np.sqrt((df.dropoff_latitude-lat_mean)**2 + (df.dropoff_longitude-long_mean)**2)

def add_time_features(df):
    timecol=pd.to_datetime(df['pickup_datetime'])
    df['pickup_year'] = timecol.dt.year
    df['pickup_hour'] = timecol.dt.hour

def get_input_matrix(df):
    return (np.column_stack((np.ones(len(df)), df.abs_diff_longitude, df.abs_diff_latitude, df.abs_dist_travel, df.up_diff_center, df.off_diff_center, df.pickup_year, df.pickup_hour)),df.fare_amount)


def train_weight(df_train):

    df_train=df_train[(df_train.pickup_latitude>39)&
                (df_train.pickup_latitude<42)&
                (df_train.pickup_longitude>-74.5)&
				(df_train.pickup_longitude<-72)&
				(df_train.dropoff_latitude>39)&
				(df_train.dropoff_latitude<42)&
				(df_train.dropoff_longitude>-74.5)&
				(df_train.dropoff_longitude<-72)&
				(df_train.fare_amount > 0)&
				(df_train.passenger_count > 0)]

	add_travel_vector_features(df_train)
	add_loc_bias_feature(df_train)
	add_time_features(df_train)

	train_X,train_y = get_input_matrix(df_train.iloc[0:df_train.shape[0]*9//10,:])
	valid_X,valid_y = get_input_matrix(df_train.iloc[df_train.shape[0]*9//10:,:])

	(w_lsr, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)
	w_lsr=w_lsr.reshape(len(w_lsr),1)
	valid_y_est = np.matmul(valid_X, w_lsr).round(decimals = 2)
	RMSE_lsr=((np.asarray(valid_y)-valid_y_est) ** 2).mean() ** .5
	return w_lsr, RMSE_lsr



chunker = pd.read_csv('../input/train.csv',chunkersize=500000)
#df_train=pd.read_csv('../input/train.csv',nrows=100000);df_train
df_test=pd.read_csv('../input/test.csv');
W=np.zeros((8,0))
E=np.zeros((1,0))
i=1

for piece in chunker:
	print('Piece: '+ str(i))
	w,e = train_weight(piece)
	W = np.c_[W,w]
	E = np.c_[E,e]
	i += 1