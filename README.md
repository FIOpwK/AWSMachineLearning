# AWSMachineLearning
Introduction to AWS Machine Learning Foundations

# Learning Objectives
- Differentiate between supervised learning and unsupervised learning.
- Identify problems that can be solved with machine learning.
- Describe commonly used algorithms including linear regression, logistic regression, and k-means.
- Describe how training and testing works.
- Evaluate the performance of a machine learning model using metrics.


# Advice From the Experts
Remember the following advice when training your model.

-  Practitioners often use machine learning frameworks that already have working implementations of models and model training algorithms. You could implement these from scratch, but you probably won't need to do so unless you’re developing new models or algorithms.

- Practitioners use a process called model selection to determine which model or models to use. The list of established models is constantly growing, and even seasoned machine learning practitioners may try many different types of models while solving a problem with machine learning.

- Hyperparameters are settings on the model which are not changed during training but can affect how quickly or how reliably the model trains, such as the number of clusters the model should identify.

- Be prepared to iterate.


# Machine Learning Tasks

## Supervised learning
In supervised learning, there are two main identifiers you will see in machine learning:

- A categorical label has a discrete set of possible values. In a machine learning problem in which you want to identify the type of flower based on a picture, you would train your model using images that have been labeled with the categories of flower you would want to identify. Futhermore, when you work with categorical labels, you often carry out classification tasks*, which are part of the supervised learning family.

- A continous (regression) label does not have a discrete set of possible values, which often means you are working with numerical data. In the snow code sales example, we are trying to predict the number* of snow cones sold. Here, our label is a number that could, in theory, be any value.


## Unsupervised learning
In unsupervised learning, *clustering* is just one example. There are many other options, such as deep learning.

- Labeled data: Supervised learning
- Unlabeled data : Unsupervised learning


## Four aspects of working with data
- *Data Collection* can be as straightfoward as running the appropriate SQL queries or as complicated as building custom web scraper applications to collect data for your project. You might even have to run a model over your data to generate needed labels. Here is a fundamental question:

    "Does the data you've collected match the machine learning task and problem you have defined?"


- *Data Inspection* The quality of your data will ultimately be the largest factor that affects how well you can expect your model to perform. As you inspect your data, look for:
    - Outliers
    - Missing or incomplete values
    - Data that needs to be transformed or processed so it's in the correct format to be used by your model

- *Summary Statistics* Models can assume how your data is structured. Now that you have some data in hand it is good best practice to check that your data is in line with the uderlying assumptions of your chosen machine learning model. 

    With many statistical tools, you can calculate things like the mean, inner-quartile range (IQR), and standard deviation. These tools can give you insight into the *scope*, *scale*, and *shape* of the dataset.


- *Data Visualization* You can use data visualization to see outliers and trends in your data and to help stakeholders understand your data.



# Five steps of Machine Learning
- Define the Problem

- Build the Dataset

- Train the Model
    Q: "What does a model training algorithm actually do?
    A: "Iteratively update model parameters to minimize some loss function"
    

- Evaluate the Model

- Use the Model
    Inference: involves
        - Generating predictions
        - Finding pattersn in your data
        - Using a trained model
        - Testing your model on data it has not seen before



# Things to think about
There are many different tools that can be used to evaluate a linear regression model. Here are a few examples:

- Mean absolute error (MAE): This is measured by taking the average of the absolute difference between the actual values and the predictions. Ideally, this difference is minimal.

- Root mean square error (RMSE): This is similar MAE, but takes a slightly modified approach so values with large error receive a higher penalty. RMSE takes the square root of the average squared difference between the prediction and the actual value.

- Coefficient of determination or R-squared (R^2): This measures how well-observed outcomes are actually predicted by the model, based on the proportion of total variation of outcomes.


# Terminology
- *Clustering* : Unsupervised learning task that helps to determine if there are any naturally occurring groupings in the data

- A *categorical label* has a discrete set of possible values, such as "is a cat" and "is not a cat."

- A *continuous (regression) label* does not have a discrete set of possible values, which means possibly an unlimited number of possibilities.

- *Discrete* : A term taken from statistics referring to an outcome taking on only a finite number of values (such as days of the week).

- A *label* refers to data that already contains the solution

- Using *unlabeled* data means you don't need to provide the model with any kind of label or solution while the model is being trained.

- *Impute* is a common term referring to different statistical tools which can be used to calculate missing values from your dataset.

- *Outliers* are data points that are significantly different from others in the same sample.


- *Model Parameters* are the configuration that changes how the model behaves. Depending on the context, you'll also hear other more specific terms used to descrive model parameters such as *weights* and *biases*. Weights, which are values that change as the model learns, are more specific to neural network.

- *Loss Function* is the measurement of how close the model is to its goal. Is used to codify the model's distance from this goal.

- *Training dataset* is the data on which the model will be trained. Most of your data will be here.

- *Test datset* is the data withheld from the model during training, which is used to test how well your model will generalize to new data.

- *Hyperparameters* are settings on the model which are not changed during training but can affect how quickly or how reliably the model trains, such as the number of clusters the model should identify.

- *Log loss* seeks to calculate how *uncertain* your model is about the predictions it is generating.

- *Model Accuracy* is the fraction of predictions a model gets right.

- *Bag of words* : A technique used to extract features from the text. It counts how many times a word appears in a document (corpus), and then transforms that information into a dataset.

- *Data vectorization*: is a process that converts non-numeric data into a numerical format so that it can be used by a machine leaning model.

- *Silhouette coefficient*: a score from -1 to 1 describing the clusters found during modeling. A score near zero indicates overlapping clusters, and scores less than zero indicate data points assigned to incorrect clusters. A score approaching 1 indicates successful identification of discrete non-overlapping clusters.

- *Stop words*: A list of words removed by natural language processing tools when building your dataset. There is no single universal list of stop words used by all-natural language processing tools.


- *Reinforcement learning* : is characterized by a continuous loop where the *agent* interacts with an *environment* and measures the consequences of its actions

- *Generative AI* : enables computers to learn the underlying pattern of a given problem and use this knowledge to generate new content from input (such as images, music, and text).

- *AR-CNN*: the autoregressive convolutional neural network generative technique uses a U-Net architecture. 

*Agent* : the piece of software you are training is called an agent. It makes decisions in an environment to reach a goal

*Environment* : the environment is the surrounding area with which the agent interacts. 

*Reward* : Feedback is given to an agent for each action it takes in a given state. This feedback is a numerical reward.

*Action* : for every state, an agent needs to take an action towards achieving its goal.

*Reward Functions* :
    - Each state on the grid is assigned a score by your reward function. You incentivize behavior that supports your car's goal of completing fast laps by giving the highest numbers to the parts of the track on which you want it to drive.

    - The reward function is the actual code you'll write to help your agent determine if the action is just took was good or bad, and how good or bad it was.

    ```"""
    def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    
    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    
    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track
    
    return float(reward)
    ```"""

*Exploration* : when a car first starts out, it *explores* by wandering in random directions. However, the more training an agents gets, the more it learns about an environment. This experience helps it become more confident about the actions it chooses (later)

*Exploitation* : means the car begins to exploit or use information from previous experiences to help it reach its goal. Different training algorithms utilize exploration and exploitation differently.

*reward graph* :
    - while training your car in the AWS DeepRacer console, your training metrics are displayed on a reward graph

    - plotting the total reward from each episode allows you to see how the model performs over time. The more rewards your car gets, the better your model performs.

*Generator*: A neural network that learns to create new data resembling the source data on which it was trained.

*Discriminator*: A neural network trained to differentiate real vs synthetic data.

*Generator loss*: Measures how far the output data deviates from the real data present in the training dataset.

*Discriminator loss*: Evaluates how well the discriminator diffentiates real vs fake data

# AWS and AWS AI devices
Introduction to machine learning with AWS and AWS AI Devices: AWS DeepComposer and AWS DeepRacer

# Learning Objectives

    - Identify AWS machine learning offerings and undertand how different services are used for different applications.

    - Describe how reinforcement learning works in the context of AWS DeepRacer.

    - Explain the fundamentals of generative AI and its applications, and describe three famous generative AI models in the context of music and AWS DeepComposer. (AR-CNN technique, GANs technique, Transformers technique)



