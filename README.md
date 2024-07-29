
# Road Intersection Detection using GANs

## Project Overview

This work introduces a novel application of Conditional Generative Adversarial
Networks (cGANs) for the task of generating steering angle representations from
road images, with a specific focus on road intersections. Leveraging road images as
condition labels for cGANs, our approach aims to produce steering angle outputs
that accurately capture the complexities of navigating through intersections. We
present a comprehensive methodology that integrates discriminative and generative
components of cGANs to effectively model the relationship between road images
and steering angles. Extensive experiments demonstrate the efficacy of our ap-
proach in generating accurate and contextually relevant steering angle predictions
across diverse road scenarios.


This project explores the use of Generative Adversarial Networks (GANs) to detect intersections on roads and autonomously steer a vehicle through driveways containing intersections. The CARLA Steering Dataset for Self Driving Cars from Kaggle is utilized to train the GAN model, aiming to enhance the vehicle's ability to navigate complex environments by identifying and responding to intersections.


Datset link : https://www.kaggle.com/datasets/zahidbooni/alltownswithweather


## Table of Contents
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Intersection Detection](#intersection-detection)
- [Results](#results)

## Data Preparation
The dataset contains labels for steering angles normalized between [-1, 1], where -1 indicates a full left turn, 1 indicates a full right turn, and 0 indicates a straight path. We utilized K-means clustering to identify clusters when the steering angle changes.

Steps:
1. **Dataset Annotation**: Label images that contain intersections manually or using computer vision techniques.
2. **Dataset Split**: Split the dataset into training, validation, and test sets.

## Model Training
### Conditional GAN (cGAN)
We modified the standard GAN architecture to a Conditional GAN (cGAN) where the generator takes both random noise and the intersection label as input.

Steps:
1. **Generator**: Our generator network is responsible for generating steering angles corresponding to input images
and noise. It takes two inputs: the input image and noise. The input image serves as the conditioning
information, guiding the generation process, while the noise introduces stochasticity and diversity
into the generated steering angles.The generator network consists of six convolutional layers designed to progressively transform
the input image and noise into a high-dimensional representation capturing essential features for
steering angle prediction. These convolutional layers employ varying filter sizes and strides to extract
hierarchical features from the input.
Notably, we concatenate a noise vector of 20 dimensions with the output of the last convolutional
layer. This step ensures that the generator leverages both input image information and additional
stochasticity from the noise, thereby enhancing the diversity and accuracy of the generated steering
angles.
In the generator, the noise input z is mapped to a fully connected hidden layer of size same as that of
the feature vector computed on the condition image to allow their summation. Then, the summation
of these vectors is mapped to another fully connected hidden layer before it is reduced to a single
output neuron,which is steering angle, completing the generation process.



<img width="542" alt="image" src="https://github.com/user-attachments/assets/9cd21d79-82f6-4730-8dd2-5acec24a7118">



3. **Discriminator**: In parallel, our architecture includes a discriminator network responsible for distinguishing between
real and generated steering angles. The discriminator network shares a similar architecture to the
generator but is conditioned on the steering angle from the dataset for each image.
Similar to the generator, the discriminator network comprises six convolutional layers tasked with
extracting features from the input images while considering the steering angle as conditioning
information. These convolutional layers are followed by dense layers, which further process the
extracted features.
The obtained feature map is then reshaped as a vector. Before each convolution layer, spatial batch
normalization is performed, and LeakyReLU activations with 0.2 negative slope are used for all
layers except for the output, which uses Tanh.
During training, the discriminator receives real steering angles from the dataset along with their
corresponding images. It computes the real loss by comparing the discriminator’s predictions with
the ground truth labels. Additionally, the discriminator receives generated steering angles from the
generator along with the generated images. It computes the fake loss by comparing the discriminator’s
predictions with the generated steering angles.
By optimizing the discriminator’s parameters based on both real and fake losses, the network learns
to accurately distinguish between real and generated steering angles, thereby guiding the training of
the generator network



<img width="542" alt="image" src="https://github.com/user-attachments/assets/837b3881-ec64-4bac-a605-a0fd9b6a071a">




## Results

<img width="769" alt="image" src="https://github.com/user-attachments/assets/1f940935-def9-4c57-a0c8-791e529a8185">



<img width="769" alt="image" src="https://github.com/user-attachments/assets/f0aece97-cb40-4df9-a518-4ec68a7b1bf5">


The results and conclusions of this work clearly support the efficacy and robustness of the proposed
conditional GAN model for steering angle prediction. Patterns in loss curves, image comparisons, as
well as evaluation metrics consistently show the model’s exceptional ability to predict steering angles
from input images. These findings represent an important advancement in the study of steering
angle prediction, utilizing the power of conditional GANs to provide a novel technique with broad
applicability in autonomous driving systems. The model’s unparalleled accuracy, authenticity; and
adaptability open up exciting possibilities for improving safety and efficiency in navigation tasks,
paving the way for the development of more resilient and intelligent autonomous vehicles.
