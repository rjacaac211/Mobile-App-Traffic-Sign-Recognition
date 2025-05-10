<div align="center">

<a href="https://github.com/rjacaac211/Mobile-App-Traffic-Sign-Recognition">
    <img src="./github/assets/Sign-sense-logo.png" alt="Sign Sense logo" title="Sign Sense logo" width="80"/>
</a>

# Sign Sense - Real-Time Mobile App Traffic Sign Recognition

[![License: Apache-2.0](https://img.shields.io/github/license/rjacaac211/Mobile-App-Traffic-Sign-Recognition?labelColor=27303D&color=0877d2)](/LICENSE)

<div align="left">
    
**Sign Sense** is a real-time mobile application for traffic sign recognition developed to enhance road safety. By leveraging Convolutional Neural Networks (CNNs) and TensorFlow Lite, the app detects and classifies traffic signs in real-time using Android devices. 

The system is designed to provide an accessible and cost-effective alternative to high-end Advanced Driver Assistance Systems (ADAS), specifically tailored to recognize traffic signs in the Philippines. It aims to increase driver awareness, reduce traffic violations, and contribute to safer roadways.

## Features

- **Real-Time Traffic Sign Detection**: Identifies traffic signs on the road using YOLOv10 for fast and efficient object detection.
- **Traffic Sign Classification**: Classifies detected traffic signs into 15 categories using a CNN-based model.
- **Mobile-Optimized Models**: Models are converted to TensorFlow Lite (TFLite) for seamless deployment on Android devices.
- **User Feedback**: Provides audio feedback and displays labeled bounding boxes for recognized traffic signs.
- **Scalable System**: Supports easy updates for detection and classification models independently.
- **Localized for the Philippines**: Recognizes 15 traffic sign classes mandated by the Philippine Land Transportation Office (LTO).

## Technologies Used

- **YOLOv10**: Used for training the traffic sign detection model.
- **TensorFlow**: Used for training the traffic sign classification model.
- **TensorFlow Lite**: Enables lightweight, efficient model deployment on Android devices.
- **Kotlin and Android Studio**: For developing the mobile application.
- **Roboflow**: For managing and annotating datasets.
- **Google Colab**: Utilized for detection (YOLO) model training with GPU resources.
- **VSCode**: Used for developing the classification model.

## Repository Structure

This repository is organized as follows:

- **`app/`**: Contains the source code for the Android application, including Gradle and Kotlin files for app development.
- **`data/`**: Includes the datasets for classification and detection tasks, organized into respective directories.
- **`recognition-models/`**: Contains the development files and Jupyter notebooks for the classification and detection models.

## Setup and Installation

To set up and run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rjacaac211/Mobile-App-Traffic-Sign-Recognition.git
   cd Mobile-App-Traffic-Sign-Recognition
   ```
2. **Set up the Android Application**
   - Open Android Studio.
   - Select **File > Open** and navigate to the `app` directory of the cloned repository.
   - Allow Android Studio to sync the Gradle files.
   - Configure the emulator or connect a physical device for testing.
   - Run the application to build and deploy it on your device.
  
## Usage

1. **Run the Android Application**:
   - Launch the app on the connected Android device or emulator.
   - Allow necessary permissions (e.g., camera access) when prompted.

2. **Real-Time Traffic Sign Recognition**:
   - Point the device's camera toward a road or traffic signs.
   - The app will detect traffic signs in real-time and display the results on-screen, including labels, bounding boxes, and audio feedback.

3. **Test the App**:
   - Use sample traffic sign images or real-world traffic signs to test the recognition functionality.
   - Review performance metrics like recognition inference time displayed in the app.

4. **Explore the Code**:
   - Visit the `recognition-models` directory for notebooks and files related to classification and detection model development.
   - Explore the `data` directory to view the datasets used for training and testing.

5. **Customize**:
   - Modify the models or datasets as needed and retrain using the provided scripts and notebooks.

</div>
