# Leaf Disease Classification System (React + FastAPI + TensorFlow)

A deep learning-based system designed to classify leaf diseases using images of leaves. The system uses TensorFlow for model training and prediction, FastAPI for the backend API, and React for the user interface.

## Features

- **Leaf Disease Classification**: Classifies leaf diseases into categories such as "Early Blight", "Late Blight", and "Healthy".
- **User-Friendly Interface**: Built with React to allow users to upload leaf images for classification and view predictions.
- **Fast API Backend**: Fast and efficient prediction using FastAPI to serve the trained TensorFlow model.
- **TensorFlow Model**: Utilizes a convolutional neural network (CNN) trained on leaf images for accurate disease classification.

## Technologies

- **Frontend**: React
- **Backend**: FastAPI
- **Model**: TensorFlow
- **Cloud Storage**: Google Cloud Storage (for storing the trained model)
- **Image Processing**: Pillow (PIL) for image handling

## Setup and Installation

### Prerequisites

- Python 3.9+
- Node.js
- TensorFlow
- FastAPI
- Google Cloud SDK (for deploying the backend)
- React (for frontend)

### Backend Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/leaf-disease-classification.git
    cd leaf-disease-classification
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate   # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the trained model from Google Cloud Storage or add it manually to the `/models` directory.

5. Run the FastAPI backend:

    ```bash
    uvicorn app:app --reload
    ```

    The backend will be running at `http://localhost:8000`.

### Frontend Setup

1. Navigate to the `frontend` directory:

    ```bash
    cd frontend
    ```

2. Install the frontend dependencies:

    ```bash
    npm install
    ```

3. Run the React app:

    ```bash
    npm start
    ```

    The React app will be available at `http://localhost:3000`.

## Usage

1. Open the React frontend in your browser.
2. Upload an image of a leaf.
3. The system will predict if the leaf is "Healthy", "Early Blight", or "Late Blight".
4. View the classification result and confidence percentage.

## Deployment

1. **Backend**: Deploy the FastAPI backend to Google Cloud Functions, or any other cloud provider.
2. **Frontend**: Deploy the React frontend to a cloud service like Vercel, Netlify, or Firebase Hosting.

## Contributing

Feel free to fork this repository, create issues, or submit pull requests if you would like to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

