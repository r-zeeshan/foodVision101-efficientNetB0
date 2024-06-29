# dataset.py
import tensorflow_datasets as tfds


def get_dataset(name, split):
    """
    Loads and returns the specified dataset.

    Parameters:
    - name (str): The name of the dataset to load.
    - split (str): The split of the dataset to load (e.g., 'train', 'test', 'validation').

    Returns:
    - tuple: A tuple containing the train data, test data, and dataset information.
    """
    (train_data, test_data), ds_info = tfds.load(name=name,
                                                 split=split,
                                                 shuffle_files=True,
                                                 as_supervised=True,
                                                 with_info=True)

    return (train_data, test_data), ds_info