
from pathlib import Path

import shutil
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

from sklearn.model_selection import train_test_split


def with_sun_as_unit_vector_in_flu(data: pd.DataFrame):
    sin_zen = np.sin(np.deg2rad(data['zen_deg']))
    cos_zen = np.cos(np.deg2rad(data['zen_deg']))
    cos_az = np.cos(np.deg2rad(data['az_deg']))
    sin_az = np.sin(np.deg2rad(data['az_deg']))
    data['sun_f'] = -sin_zen*sin_az
    data['sun_l'] = sin_zen*cos_az
    data['sun_u'] = cos_zen 


def get_image_names(image_dir: Path, left=True):
    if left:
        return sorted([str(x) for x in image_dir.glob('*left*.pgm')])
    return sorted([str(x) for x in image_dir.glob('*.pgm')])


def extract_image_index(image_name):
    """
    Takes an image name like grey-rectified-left-000002.pgm
    and returns the index 2
    """
    return int(image_name.split('-')[-1].split('.')[0])


def generate_image_name(image_base_dir: str, image_index: int, left=True):
    """
    Take an index eg 2 and returns a name like grey-rectified-left-000002.pgm
    """
    image_name = f'{image_index:06d}.pgm'
    if left:
        return f'{image_base_dir}/grey-rectified-left-{image_name}'
    return f'{image_base_dir}/grey-rectified-right-{image_name}'


def get_max_min_index(image_names):
    max_index = 0
    min_index = 100000000
    for image_name in image_names:
        image_index = extract_image_index(image_name)
        if image_index > max_index:
            max_index = image_index
        if image_index < min_index:
            min_index = image_index
    return max_index, min_index


def build_image_index_lookup(base_dir: Path):
    """
    Looks at the image folders in the base_dir and builds a lookup,
    ignores the zip files
    """
    image_index_lookup = {}
    for image_dir in base_dir.glob('*'):
        if image_dir.name.endswith('.zip'):
            continue
        image_names = get_image_names(image_dir)
        max_index, min_index = get_max_min_index(image_names)
        image_index_lookup[image_dir.name] = (min_index, max_index)
    def lookup_index(index: int):
        for base_image_name, (min_index, max_index) in image_index_lookup.items():
            if min_index <= index <= max_index:
                return generate_image_name(base_dir / base_image_name, index)
        raise ValueError(f'Index {index} not found in lookup')
    return lookup_index


def generate_enriched_sun_data(max_index=100_000):
    """
    Enriches the sun-sensor-sampled.txt data with the image name and the sun unit vector
    and then saves it as a csv file
    """
    # First load the sun sampled data
    data = pd.read_csv(
        'example_dataset/raw_data/sun-sensor-sampled.txt', 
        sep="\s+|\t+|\s+\t+|\t+\s+", 
        header=None, 
        names=['image_ind', 'az_deg', 'zen_deg', 'time_offset_s']
    )
    data = data[data['image_ind'] <= max_index]
    # Enrich it with the sun unit vector
    with_sun_as_unit_vector_in_flu(data)
    # Load the image index lookup
    lookup_function = build_image_index_lookup(Path('/home/hugo/datasets/devon_island/grey-rectified-512x384/'))
    # Add the image name to the data
    data['image_name'] = data['image_ind'].apply(lookup_function)
    print(data)
    # Save the data as a csv file
    # Create the output directory if it doesn't exist
    output_dir = Path('example_dataset/enriched_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / 'sun-sensor-sampled.csv')


def copy_sampled_data(max_index=100_000):
    """
    Copies images from their original image locations to a new directory example_dataset/enriched_data/sampled
    does not move the images, just copies them
    """
    data = pd.read_csv('example_dataset/enriched_data/sun-sensor-sampled.csv')
    data = data[data['image_ind'] <= max_index]
    for image_name in data['image_name']:
        image_name = Path(image_name)
        new_image_name = Path(f'example_dataset/enriched_data/sampled/{image_name.name}')
        new_image_name.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(image_name), str(new_image_name))


def update_image_names_to_sampled():
    """
    Updates the image names in the sun-sensor-sampled.csv file to point to the sampled images
    with a relative path
    """
    data = pd.read_csv('example_dataset/enriched_data/sun-sensor-sampled.csv')
    data['image_name'] =data['image_name'].apply(lambda x: "example_dataset/enriched_data/sampled/" + Path(x).name)
    data.to_csv('example_dataset/enriched_data/sun-sensor-sampled.csv')

def analyse_dataset_balance(csv_file='example_dataset/enriched_data/sun-sensor-sampled.csv'):
    """
    Plots graphs to show the balance of the dataset, specifically looking at
    the distribution of the azimuth and zenith angles
    """
    data = pd.read_csv(csv_file)
    print(f'Total number of samples: {len(data)}')
    # Plot the azimuth distribution
    plt.figure()
    plt.hist(data['az_deg'], bins=36)
    plt.title(f'Azimuth distribution\n{csv_file}')
    # Plot the zenith distribution
    plt.figure()
    plt.hist(data['zen_deg'], bins=36)
    plt.title(f'Zenith distribution\n{csv_file}')
    plt.show()


def balance_dataset_by_azimuth_angle(n_bins: int = 36):
    """
    Loads the sun sensor csv file and then balances the dataset by azimuth angle
    Writes a sampled csv file to example_dataset/enriched_data/sampled_balanced.csv
    this csv file has a balanced distribution of azimuth angles in n_bins bins
    """
    data = pd.read_csv('example_dataset/enriched_data/sun-sensor-sampled-combined.csv')
    # Bin the azimuth angles into n_bins bins
    data['az_deg_bin'] = pd.cut(data['az_deg'], n_bins, labels=False)
    # Count the number of samples in each bin
    bin_counts = data.groupby('az_deg_bin').count()['az_deg']
    # Find the bin with the least samples
    min_bin = bin_counts.idxmin()
    # Find the number of samples in the min bin
    min_bin_count = bin_counts[min_bin]
    # Create a new dataframe with the same number of samples in each bin
    new_data = []
    for bin_index in range(n_bins):
        # Get the data for this bin
        bin_data = data[data['az_deg_bin'] == bin_index]
        # Sample the data to the minimum bin count
        bin_data = bin_data.sample(min_bin_count)
        # Add the sampled data to the new dataframe
        for bd in bin_data.iterrows():
            new_data.append(bd[1])
    new_data_pd = pd.DataFrame(new_data)
    print(new_data_pd)
    # Save the new dataframe as a csv file
    new_data_pd.to_csv('example_dataset/enriched_data/sampled_balanced.csv')


def copy_sampled_with_flip():
    """
    Copy the sampled image directory to a new directory and save them with a horizontal flip
    Adds the suffix _flip to the image name
    """
    sampled_image_dir = Path('example_dataset/enriched_data/sampled')
    sampled_image_dir_flip = Path('example_dataset/enriched_data/sampled_flip')
    sampled_image_dir_flip.mkdir(parents=True, exist_ok=True)
    for image_name in sampled_image_dir.glob('*.pgm'):
        image_name_flip = sampled_image_dir_flip / f'{image_name.stem}_flip.pgm'
        image_name_flip.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(str(image_name))
        image_flip = ImageOps.mirror(image)
        image_flip.save(str(image_name_flip))


def map_angle_degs_to_180_minus_180(angle_degs_array):
    """
    Maps an array of angles in degrees to the range -180 to 180
    even if the angle is outside this range
    eg. -190 -> 170
    eg. 190 -> -170
    """
    angle_degs_array = np.array(angle_degs_array)
    angle_degs_array = angle_degs_array % 360
    angle_degs_array[angle_degs_array > 180] -= 360
    return angle_degs_array


def generate_flipped_sun_sensor_csv():
    """
    Reads the sun sensor sampled csv file and flips the sun unit vector in the left direction
    Adds the suffix _flip to the image name. Adds _flip to the sampled in the image name
    """
    data = pd.read_csv('example_dataset/enriched_data/sun-sensor-sampled.csv')
    data['sun_f'] = data['sun_f']
    data['sun_l'] = -data['sun_l']
    data['sun_u'] = data['sun_u']
    data['az_deg'] = map_angle_degs_to_180_minus_180(data['az_deg'] - 180)
    data['image_name'] = data['image_name'].apply(lambda x: x.replace('sampled', 'sampled_flip'))
    data['image_name'] = data['image_name'].apply(lambda x: x.replace('.pgm', '_flip.pgm'))
    data.to_csv('example_dataset/enriched_data/sun-sensor-sampled-flipped.csv')


def combine_with_flipped_data():
    """
    Combines the sampled csv file with the flipped sampled csv file
    Adds 100_000 to the image index of the flipped data
    """
    sampled_data = pd.read_csv('example_dataset/enriched_data/sun-sensor-sampled.csv')
    sampled_data_flip = pd.read_csv('example_dataset/enriched_data/sun-sensor-sampled-flipped.csv')
    sampled_data_flip['image_ind'] += 100_000
    combined_data = pd.concat([sampled_data, sampled_data_flip])
    combined_data.to_csv('example_dataset/enriched_data/sun-sensor-sampled-combined.csv')


def generate_train_and_test_data():
    """
    Generates the train and test data
    Reads the balanced csv file and splits it into train and test data, and saves these as train.csv and test.csv
    Then symlinks images from the sampled_flip directory and sampled directory to the train and test directories
    """
    data = pd.read_csv('example_dataset/enriched_data/sampled_balanced.csv')
    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size=0.2)
    # Create the train and test directories
    train_dir = Path('example_dataset/train/images')
    test_dir = Path('example_dataset/test/images')
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    # Create symlinks to the images in the train and test directories
    for image_name in train_data['image_name']:
        image_name = Path(image_name)
        image_name_train = train_dir / image_name.name
        image_name_train.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(Path.cwd() /image_name, image_name_train)
        except:
            print(f'Failed to symlink {image_name} to {image_name_train}')
    for image_name in test_data['image_name']:
        image_name = Path(image_name)
        image_name_test = test_dir / image_name.name
        image_name_test.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(Path.cwd() / image_name, image_name_test)
        except:
            print(f'Failed to symlink {image_name} to {image_name_test}')
    # Add the test and train image names back to the data
    # test_dir / image_name.name
    train_data['image_name'] = train_data['image_name'].apply(lambda x: str(train_dir / Path(x).name))
    test_data['image_name'] = test_data['image_name'].apply(lambda x: str(test_dir / Path(x).name))
    # Cut the train and test data down to just the image name and sun unit vector
    train_data = train_data[['image_name', 'sun_f', 'sun_l', 'sun_u']]
    test_data = test_data[['image_name', 'sun_f', 'sun_l', 'sun_u']]
    # Save the train and test data as csv files
    train_data.to_csv('example_dataset/train.csv', index=False)
    test_data.to_csv('example_dataset/test.csv', index=False)


def check_for_duplicate_image_names():
    """
    Checks for duplicate image names in the data
    """
    data = pd.read_csv('example_dataset/enriched_data/sampled_balanced.csv')
    image_names = data['image_name']
    image_names = [Path(x).name for x in image_names]
    image_names = np.array(image_names)
    unique, counts = np.unique(image_names, return_counts=True)
    print(unique)
    print(counts)
    print(np.sum(counts > 1))


if __name__ == '__main__':
    generate_enriched_sun_data()
    # copy_sampled_data()
    update_image_names_to_sampled()
    # copy_sampled_with_flip()
    generate_flipped_sun_sensor_csv()
    combine_with_flipped_data()
    analyse_dataset_balance()
    analyse_dataset_balance('example_dataset/enriched_data/sun-sensor-sampled-combined.csv')
    balance_dataset_by_azimuth_angle(n_bins=36)
    analyse_dataset_balance('example_dataset/enriched_data/sampled_balanced.csv')
    generate_train_and_test_data()
    # check_for_duplicate_image_names()
