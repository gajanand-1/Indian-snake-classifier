🐍 Indian Snake Species Classifier
This project is a web application that uses a deep learning model to classify 13 different species of snakes commonly found in the Indian subcontinent. The app is built with Streamlit and is powered by a fine-tuned ResNet50v2 model implemented in PyTorch.

🚀 Live Demo
You can access and try the live application here:

➡️ https://indian-snake-classifier-tr7bqqkwbbfcynmy8shwhk.streamlit.app/

📸 Application Screenshot
Here is a look at the application's user interface.

✨ Key Features
Two Input Methods: Users can either upload an image file directly from their device or paste a URL to an image on the web.

Focused Classification: The model is specifically trained to identify snake species prevalent in the Indian region.

Deep Learning Model: Utilizes a ResNet50v2 model, fine-tuned for high accuracy on this specific task.

Real-time Prediction: Provides a prediction for the snake species along with a confidence score.

Interactive Interface: A clean and user-friendly interface built with Streamlit for a smooth user experience.

🤖 Model & Dataset Details
The model was trained with a strong focus on data quality to ensure robust performance.

Model Architecture: ResNet50v2

Framework: PyTorch

Dataset: A custom dataset was carefully curated by combining and filtering two Kaggle datasets ("Snake Dataset" and "Snakes Species Dataset"). The final dataset was tailored to include 13 snake species typically found in India.

Data Preprocessing: Significant effort was put into cleaning the data:

Noise Removal: Irrelevant and low-quality images were removed.

Re-labeling: Incorrectly classified images were corrected.

Data Augmentation: Techniques were applied to increase the diversity of the training set and prevent overfitting.

Training Details:

Trained on 2,167 images.

Validated on 323 images.

Performance:

Training Accuracy: 93.95%

Validation Accuracy: 86.38%
