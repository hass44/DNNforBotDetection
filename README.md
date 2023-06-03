# Twitter Bot Detection using Deep Learning

This repository contains code and resources for detecting Twitter bots using deep learning techniques. The project aims to identify and classify Twitter accounts as bots or non-bots based on various features extracted from their profiles and activities.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributors](#contributors)

## Introduction

With the increasing presence of bots on social media platforms, it has become important to develop effective methods for detecting and identifying these automated accounts. This project focuses on Twitter bot detection using a deep learning approach. The model is trained on a dataset of bot and non-bot Twitter accounts, and it learns to classify new accounts based on their features.

## Dataset

The dataset used for training and evaluation is cresci-2017, a combination of bot and non-bot Twitter accounts. It includes features such as username, age, location, verification status, tweet count, follower count, and more. The dataset is divided into training and test sets to train the model and evaluate its performance. Publicily available at [here](https://botometer.osome.iu.edu/bot-repository/datasets.html)

## Model Architecture

The model architecture used for Twitter bot detection is a deep neural network. It consists of multiple dense layers with ReLU activation and a final sigmoid activation layer for binary classification. The model takes various account features as input and predicts the probability of an account being a bot.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/twitter-bot-detection.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:

   ```bash
   python app.py
   ```

4. Access the web application in your browser at `http://localhost:5000/home`.

## Evaluation

The model is evaluated on the test dataset using metrics such as precision, recall, F1 score, and ROC/AUC score. These metrics provide an assessment of the model's performance in correctly classifying bot and non-bot accounts. Flask app is built to test for real world data.


## Results
###Account-Level Classification
The model achieves high precision, recall, F1 score, and ROC/AUC score, indicating its effectiveness in detecting Twitter bots. However, it's important to note that the performance may vary depending on the dataset and real-world scenarios.

| Metric         | Score                |
|----------------|----------------------|
| Recall Score   | 0.9756345177664975   |
| F1 Score       | 0.9821154828819622   |
| ROC/AUC Score  | 0.9962218266786795   |

###Tweet-Level Classification
The model achieves good precision, recall, F1 score, and ROC/AUC score, indicating its effectiveness in detecting Twitter bots. However, it's important to note that the performance may vary depending on the dataset and real-world scenarios.

| Metric         | Score                |
|----------------|----------------------|
| Recall Score   | 0.8872471207100318   |
| F1 Score       | 0.834560110730113    |
| ROC/AUC Score  | 0.8468545091803743   |

## Contributors

- [Muhammad Hassaan Ibrahim](https://github.com/hass44)
- [Ahmad Faraz Sheikh](https://www.linkedin.com/in/ahmad-faraz-sheikh)
