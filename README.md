
# Road Intersection Detection using GANs

## Project Overview

This project introduces a novel application of Conditional Generative Adversarial Networks (cGANs) for generating steering angle representations from road images, with a specific focus on road intersections. Our approach uses road images as condition labels for cGANs to produce steering angle outputs that accurately capture the complexities of navigating through intersections. The methodology integrates the discriminative and generative components of cGANs to model the relationship between road images and steering angles effectively. Extensive experiments demonstrate the efficacy of this approach in generating accurate and contextually relevant steering angle predictions across diverse road scenarios.

The project explores using Generative Adversarial Networks (GANs) to detect intersections and autonomously steer a vehicle through driveways containing intersections. The CARLA Steering Dataset for Self-Driving Cars from Kaggle is used to train the GAN model, aiming to enhance the vehicle's ability to navigate complex environments by identifying and responding to intersections.

Dataset link: [CARLA Steering Dataset](https://www.kaggle.com/datasets/zahidbooni/alltownswithweather)

## Table of Contents
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Intersection Detection](#intersection-detection)
- [Results](#results)

## Data Preparation

The dataset contains labels for steering angles normalized between [-1, 1], where:
- **-1** indicates a full left turn
- **1** indicates a full right turn
- **0** indicates a straight path

### Steps:
1. **Dataset Annotation**: Label images that contain intersections manually or using computer vision techniques.
2. **Dataset Split**: Split the dataset into training, validation, and test sets.

## Model Training

### Conditional GAN (cGAN)

We modified the standard GAN architecture to a Conditional GAN (cGAN), where the generator takes both random noise and the intersection label as input.

#### Generator

The generator network generates steering angles corresponding to input images and noise. It takes two inputs: the input image and noise. The input image serves as the conditioning information, while the noise introduces stochasticity and diversity.

- **Architecture**: The generator consists of six convolutional layers designed to progressively transform the input image and noise into a high-dimensional representation for steering angle prediction. The noise vector (20 dimensions) is concatenated with the output of the last convolutional layer to enhance the diversity and accuracy of the generated steering angles.

- **Output**: The noise input \( z \) is mapped to a fully connected hidden layer, combined with the feature vector computed from the conditioning image, and then mapped to another fully connected hidden layer before being reduced to a single output neuron, which represents the steering angle.

![Generator Architecture](https://github.com/user-attachments/assets/9cd21d79-82f6-4730-8dd2-5acec24a7118)

#### Discriminator

The discriminator network distinguishes between real and generated steering angles. It shares a similar architecture with the generator but is conditioned on the steering angle from the dataset for each image.

- **Architecture**: The discriminator comprises six convolutional layers, followed by dense layers. It uses spatial batch normalization and LeakyReLU activations, except for the output layer, which uses Tanh.

- **Training**: The discriminator receives real steering angles with their corresponding images and generated steering angles with generated images. It computes real and fake losses, guiding the generator network's training.

![Discriminator Architecture](https://github.com/user-attachments/assets/837b3881-ec64-4bac-a605-a0fd9b6a071a)

## Results

The results and conclusions support the efficacy and robustness of the proposed conditional GAN model for steering angle prediction. The model demonstrates exceptional ability in predicting steering angles from input images, as shown by the loss curves, image comparisons, and evaluation metrics.

- **Accuracy**: The model's accuracy, authenticity, and adaptability enhance navigation tasks, paving the way for developing more resilient and intelligent autonomous vehicles.

![Results](https://github.com/user-attachments/assets/1f940935-def9-4c57-a0c8-791e529a8185)

![Results](https://github.com/user-attachments/assets/f0aece97-cb40-4df9-a518-4ec68a7b1bf5)

