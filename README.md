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