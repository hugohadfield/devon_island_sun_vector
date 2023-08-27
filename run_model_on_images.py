
from typing import Tuple
from pathlib import Path

import tqdm


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from network import load_model, RegressionTaskData, calculate_angles_between_sun_vectors
from generate_devon_island_dataset import (
    get_image_names, extract_image_index, get_max_min_index, with_sun_as_unit_vector_in_flu
)


def path_target_generator(image_dir: Path):
    """
    This function returns a map that is back to the image path
    """
    image_names = get_image_names(image_dir)
    return {image_name: image_name for image_name in image_names}


def run_model_on_images(model_name: Path, image_dir: Path, image_size: Tuple[int, int, int] = (1, 100, 100)):
    """
    This function runs the model on the images in the image_dir and saves the results to a csv file
    It runs the model on the GPU
    """
    if image_size[0] == 1:
        grayscale = True
    else:
        grayscale = False
    assert image_size[1] == image_size[2], 'Image size must be square'
    resize_size = image_size[1]
    regression_task = RegressionTaskData(grayscale=grayscale, resize_size=resize_size)

    # First we load the model
    model = load_model(image_size=image_size, filename=model_name)
    # model.eval()

    # We need to move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model.to(device)

    # Delete the existing results file if it exists
    if Path('results.csv').exists():
        Path('results.csv').unlink()

    # Run the model on the data and store the results in a csv file
    with torch.no_grad():
        evaluation_loader = regression_task.make_dataloader(image_dir, path_target_generator(image_dir / 'images'))
        for inputs, image_path_batch in tqdm.tqdm(evaluation_loader):

            n_mc_evals = 20
            output_totals = np.zeros((len(inputs), 3))
            outputs_vars = np.zeros((len(inputs), 3))
            for i in range(n_mc_evals):
                outputs = model(inputs.to(device))
                outputs_np = outputs.detach().cpu().numpy()
                output_totals += outputs_np
                outputs_vars += outputs_np**2
            outputs_mean = output_totals/(1 + n_mc_evals)
            outputs_vars = outputs_vars/(1 + n_mc_evals) - outputs_mean**2

            # Store the result of this batch in a csv file and each of the image names
            with open('results.csv', 'a') as f:
                for op_mu, op_var, image_name in zip(outputs_mean, outputs_vars, image_path_batch):
                    f.write(f'{image_name},{op_mu[0]},{op_mu[1]},{op_mu[2]},{op_var[0]},{op_var[1]},{op_var[2]}\n')


def symlink_images_to_evalutation_folder(image_dir: Path):
    """
    Symlinks the images in image_dir to the evaluation folder
    The evaluation folder is the folder that the model will run on, it is example_dataset/evaluation
    """
    evaluation_dir = Path('example_dataset/evaluation/images')
    if not evaluation_dir.exists():
        evaluation_dir.mkdir(exist_ok=True, parents=True)
    for image_name in get_image_names(image_dir):
        if not (evaluation_dir / Path(image_name).name).exists():
            (evaluation_dir / Path(image_name).name).symlink_to(image_name)


def load_sequence_sun_vector_data(max_ind: int, min_ind: int) -> pd.DataFrame:
    data = pd.read_csv('example_dataset/raw_data/sun-sensor-sampled.txt', sep="\s+|\t+|\s+\t+|\t+\s+", header=None, names=['image_ind', 'az_deg', 'zen_deg', 'time_offset_s'])
    with_sun_as_unit_vector_in_flu(data)
    data = data[data['image_ind'] <= max_ind]
    data = data[data['image_ind'] >= min_ind]
    return data


def plot_results_csv():
    """
    This function plots the results csv file
    """
    results = pd.read_csv('results.csv', header=None, names=['image_name', 'sun_f', 'sun_l', 'sun_u', 'sun_f_var', 'sun_l_var', 'sun_u_var'])
    results['image_ind'] = results['image_name'].apply(extract_image_index)
    results = results.set_index('image_ind')
    results = results.sort_index()
    plt.figure()
    results['sun_f'].plot()
    results['sun_l'].plot()
    results['sun_u'].plot()
    # Add a legend
    plt.legend(['sun_f', 'sun_l', 'sun_u'])
    # Calculate the azimuth and zenith angles from the sun vector
    results['az_deg'] = np.rad2deg(np.arctan2(results['sun_f'], results['sun_l']))
    results['zen_deg'] = np.rad2deg(np.arccos(results['sun_u']))
    plt.figure()
    results['az_deg'].plot()
    max_ind, min_ind = get_max_min_index(results['image_name'])
    sv_data = load_sequence_sun_vector_data(max_ind, min_ind)
    plt.plot(sv_data['image_ind'], np.rad2deg(np.arctan2(sv_data['sun_f'], sv_data['sun_l'])))
    # Add more minor ticks to the x axis
    plt.minorticks_on()
    plt.grid(which='both')
    # Add a legend
    plt.legend(['predicted', 'actual'])

    plt.figure()
    plt.plot(results['sun_f_var'])
    plt.plot(results['sun_l_var'])
    # plt.plot(results['sun_u_var'])
    plt.legend(['sun_f_var', 'sun_l_var'])
    plt.minorticks_on()
    plt.grid(which='both')

    plt.figure()
    plt.plot(np.sqrt(results['sun_f']**2 + results['sun_l']**2 + results['sun_u']**2))
    plt.title('Norm of sun vector')
    plt.minorticks_on()
    plt.grid(which='both')

    plt.figure()
    plt.plot(results['sun_f_var'] + results['sun_l_var'])
    plt.title('Approx angle var')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.show()




if __name__ == '__main__':
    # Remove the evaluation directory
    if Path('example_dataset/evaluation/images').exists():
        for file in Path('example_dataset/evaluation/images').glob('*'):
            file.unlink()
        Path('example_dataset/evaluation/images').rmdir()
    original_image_dir = Path('/home/hugo/datasets/devon_island/grey-rectified-512x384/grey-rectified-512x384-s22/')
    symlink_images_to_evalutation_folder(original_image_dir)
    run_model_on_images(
        '1_100_100.pth', 
        Path('example_dataset/evaluation'),
        image_size=(1, 100, 100)
    )
    plot_results_csv()
