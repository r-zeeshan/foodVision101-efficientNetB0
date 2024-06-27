from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from config import CHECKPOINT_PATH
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


def get_callbacks():
    """
    Returns a list of callbacks for training a model.
    
    Returns:
        list: A list of callbacks.
    """
    return [early_stopping(),
            reduce_lr(),
            model_checkpoint(path=CHECKPOINT_PATH)]
