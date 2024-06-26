from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime


def model_checkpoint(path='modelCheckPoints/checkpoint.ckpt'):
    """
    Create a ModelCheckpoint callback.

    Args:
        path (str): The path to save the model checkpoints.

    Returns:
        ModelCheckpoint: The ModelCheckpoint callback object.

    """
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



def tensorboard(dir_name='food_vision', exp_name='efficentnetb3'):
    """
    Creates a TensorBoard callback instand to store log files.
    Stores log files with the filepath:
        "dir_name/experiment_name/current_datetime/"
    Args:
        dir_name: target directory to store TensorBoard log files
        experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    """

    log_dir = dir_name + '/' + exp_name + '/' + datetime.dateime.now().strftime("%Y%m%d-%H%M%S")        
    return TensorBoard(log_dir=log_dir)


def get_callbacks():
    return [early_stopping(),
            reduce_lr(),
            model_checkpoint(),
            tensorboard()]



    
    


