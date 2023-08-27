
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def with_sun_as_unit_vector_in_flu(data: pd.DataFrame):
    sin_zen = np.sin(np.deg2rad(data['zen']))
    cos_zen = np.cos(np.deg2rad(data['zen']))
    cos_az = np.cos(np.deg2rad(data['az']))
    sin_az = np.sin(np.deg2rad(data['az']))
    data['sun_f'] = -sin_zen*sin_az
    data['sun_l'] = sin_zen*cos_az
    data['sun_u'] = cos_zen


if __name__ == '__main__':
    data = pd.read_csv('example_dataset/raw_data/sun-sensor-sampled.txt', sep="\s+|\t+|\s+\t+|\t+\s+", header=None, names=['image_ind', 'az', 'zen', 'time_offset_s'])
    with_sun_as_unit_vector_in_flu(data)
    print(data)

    plt.plot(data['az'])
    # plt.plot(data['sun_l'])
    # plt.plot(data['sun_u'])
    # plt.hist(data['az'], bins=100)
    # plt.plot(data['image_ind'], data['time_offset_s'], '-')
    plt.show()
