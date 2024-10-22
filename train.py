# train.py
from dataset import get_dataset
from preprocess import batch_data
from architecture import create_model
from callbacks import get_callbacks
from config import *
import pickle
import os


### Loading and preprocessing the Dataset
(train_data, test_data), ds_info = get_dataset(name=DATASET_NAME, split=DATASET_SPLIT)

train_data, test_data = batch_data(train_data=train_data, test_data=test_data)

class_names = ds_info.features['label'].names


### Creating the model
model = create_model(base=BASE,shape=SHAPE, pooling=POOLING, activation=ACTIVATION, class_names=class_names, loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)


### Setting up callbacks
callbacks = get_callbacks()


model.load_weights(CHECKPOINT_PATH)

### Training the Model
history = model.fit(train_data,
                    epochs=20,
                    validation_data=test_data,
                    callbacks=callbacks)


print("Evaluating the Model...")
model.evaluate(test_data)
print("Evaluation Complete...")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)


### Saving the Model and the history
print("\nSaving the Model...")
model.save(MODEL_PATH)

with open(HISTORY_PATH, 'wb') as file:
    pickle.dump(history, file)