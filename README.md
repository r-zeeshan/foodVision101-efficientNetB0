# FoodVision101

## Overview
**FoodVision101** is a machine learning application designed to classify images of food into one of 101 categories. This project utilizes a fine-tuned EfficientNetB0 model trained on the Food101 dataset. The web interface is built using Streamlit, allowing users to upload an image or enter an image URL for classification.

## Features
- **Image Upload**: Users can upload images in JPG or JPEG format.
- **URL Input**: Users can provide an image URL or a base64 encoded image.
- **Top Predictions**: Displays the top 10 predicted categories with probabilities.
- **Visualizations**: Uses Plotly to create interactive bar plots of the top predictions.

## Project Structure
├── app.py # Main Streamlit application
├── architecture.py # Model architecture setup
├── callbacks.py # Callbacks for model training
├── config.py # Configuration variables
├── dataset.py # Dataset loading functions
├── preprocess.py # Data preprocessing functions
├── train.py # Model training script
├── utils.py # Utility functions for prediction and plotting
└── requirements.txt # Required Python packages


## How to Run
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/FoodVision101.git
    cd FoodVision101
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## Model Evaluation
The model was evaluated on the test dataset, achieving significant accuracy. Below are the key performance metrics:
- **Accuracy**: [Include your model's accuracy here]
- **Precision, Recall, F1-Score**: Detailed class-wise performance metrics are visualized in the app.

## Demo
![Demo GIF](https://foodvision101.streamlit.app/)

## Acknowledgments
- This project uses the [Food101 dataset](https://www.tensorflow.org/datasets/catalog/food101).
- The model architecture is based on [EfficientNetB0](https://arxiv.org/abs/1905.11946).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or inquiries, please contact [your-email@example.com](mailto:fiverr.zeeshanrajpootkr@gmail.com).
