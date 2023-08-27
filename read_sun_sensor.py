
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def with_sun_as_unit_vector_in_flu(data: pd.DataFrame):
    sin_zen = np.sin(np.deg2rad(data['zen_deg']))
    cos_zen = np.cos(np.deg2rad(data['zen_deg']))
    cos_az = np.cos(np.deg2rad(data['az_deg']))
    sin_az = np.sin(np.deg2rad(data['az_deg']))
    data['sun_f'] = -sin_zen*sin_az
    data['sun_l'] = sin_zen*cos_az
    data['sun_u'] = cos_zen
    data['time'] = pd.to_datetime(data['time'])


if __name__ == '__main__':
    data = pd.read_csv('example_dataset/raw_data/sun-sensor.txt', sep="\s+|\t+|\s+\t+|\t+\s+", header=None, names=['time', 'az_deg', 'zen_deg'])
    with_sun_as_unit_vector_in_flu(data)
    print(data)

    # plt.plot(data['sun_f'])
    # plt.plot(data['sun_l'])
    # plt.plot(data['sun_u'])
    # plt.hist(data['az'], bins=100)
    plt.plot(data['time'], data['az_deg'], '-')
    plt.show()
