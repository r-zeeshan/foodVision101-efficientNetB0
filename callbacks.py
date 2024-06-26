from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from config import CHECKPOINT_PATH, TENSORBOARD_DIR_NAME, TENSORBOARD_EXP_NAME
import datetime
import os


def model_checkpoint(path):
    """
    Create a ModelCheckpoint callback.

    Args:
        path (str): The path to save the model checkpoints.

    Returns:
        ModelCheckpoint: The ModelCheckpoint callback object.

    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    return ModelCheckpoint(path,
                            monitor='val_accuracy',
                            save_best_only=True,
                            save_weights_only=True,
                            verbose=0,
                            save_freq='epoch')


    
def early_stopping():
    """
    Create an EarlyStopping callback.

    Returns:
        EarlyStopping: The EarlyStopping callback object.
    """
    return EarlyStopping(monitor='val_loss', patience=3)


def reduce_lr():
    """
    Create a ReduceLROnPlateau callback.

    Returns:
        ReduceLROnPlateau: The ReduceLROnPlateau callback object.
    """
    return ReduceLROnPlateau(monitor='val_loss',
                                factor=0.2,
                                patience=2,
                                verbose=1,
                                min_lr=1e-7)



def tensorboard(dir_name, exp_name):
    """
    Creates a TensorBoard callback instand to store log files.
    Stores log files with the filepath:
        "dir_name/experiment_name/current_datetime/"
    Args:
        dir_name: target directory to store TensorBoard log files
        experiment_name: name of experiment directory (e.g. efficientnetb3)
    """

    log_dir = dir_name + '/' + exp_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")        
    return TensorBoard(log_dir=log_dir)


def get_callbacks():
    """
    Returns a list of callbacks for training a model.
    
    Returns:
        list: A list of callbacks.
    """
    return [early_stopping(),
            reduce_lr(),
            model_checkpoint(path=CHECKPOINT_PATH),
            tensorboard(dir_name=TENSORBOARD_DIR_NAME, exp_name=TENSORBOARD_EXP_NAME)]
