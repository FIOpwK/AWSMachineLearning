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

- In supervised learning, every training sample from the dataset has a corresponding label or output value associated with it. As a result, the algorithm learns to predict labels or output values. We will explore this in-depth in this lesson


## Unsupervised learning
In unsupervised learning, *clustering* is just one example. There are many other options, such as deep learning.

- Labeled data: Supervised learning
- Unlabeled data : Unsupervised learning

- In unsupervised learning, there are no labels for the training data. A machine learning algorithm tries to learn the underlying patterns or distributions that govern the data.

## Reinforcement learning
In reinforcement learning, the algorithm figures out which actions to take in a situation to maximize a reward (in the form of a number) on the way to reaching a specific goal. This is a completely different approach than supervised and unsupervised learning. 

## How does machine learning differ from traditional programming-based approaches?
    - In traditional problem-solving with software, a person analyzes a problem and engineers a solution in code to solve that problem. For many real-world problems, this process can be laborious (or even impossible) because a correct solution would need to consider a vast number of edge cases.

    - In machine learning, the problem solver abstracts away part of their solution as a flexible component called a *model*, and uses a special program called a *model training algorithm* to adjust that model to real-world data. The result is a trained model which can be used to predict outcomes that are not part of the data set used to train it.

    In a way, machine learning automates some of the statistical reasoning and pattern-matching the problem solver would traditionally do.  The overall goal is to use a *model* created by a *model training algorithm* to generate predictions or find patterns in data that can be used to solve a problem. (statistics, applied math, computer science) <fields that influence machine learning>

    - Nearly all tasks solved with machine learning involve three primary components:
        - A machine learning model
        - A model training algorithm
        - A model inference algorithm

# What are machine learning models?

A machine learning model is a block of code or framework that can be modified to solve different but related problems based on the data provided. (A model is an extremely generic program made specific by the data used to train it. It is used to solve different problems)

## How are model training algorithms used to train a model?

This process is called *model training*. Model training algorithms work through an interactive process:
    - Think about the changes that need to be made. A model training algorithm uses the model to process data and then compares the results against some end goal.


    - Make those changes. A model training algorithm gently nudges specific parts of the model in a direction that brings the model closer to achieving the goal.

    - Repeat. By iterating over these steps over and over, you get closer and closer to what you want until you determine that you're close enough that you can stop.


## Model Inference: Using your trained model
We are ready to use the model inference algorithm to generate predictions using the trained model. This process is referred to as model inference.

# Quiz: What is Machine Learning?
    Which of the following are the primary components used in machine learning?
        - A model integrity algorithm
        o A machine learning model
        o A model training algorithm
        - A preparation algorithm
        o A model inference algorithm

    Is it true or false that you always need to have an idea of what you're making when you're handling your raw data?
     - True
     o False

    What are the three common components of machine learning. 


# Introduction to the Five Machine Learning Steps

## Major steps in machine learning process
Regardless of the specific model or training algorithm used, machine learning practitioners practice a common workflow to accomplish machine learning tasks. These steps are iterative. 
    - Step One: Define the problem
    - Step Two: Build the dataset
    - Step Three: Train the model
    - Step Four: Evaluate the model
    - Step Five: Inference (Use the model)


## How do you Start a Machine Learning Task?
    ## Define a very specific task
        - Does adding a high charge fee increase sales?
        - Does adding a low charge fee increse sales?

    ## Identify the machine learning task we might use to solve this problem
        - helps to better understand the data for the project

## What is a Machine Learning Task?
All model training algorithms, and the models themselves, take data as their input. Their outputs can be very different and are classified into a few different groups based on the *task* they are designed to solve. Often, we use the kind of data required to train a model as part of defining the machine learning task.

- Supervised learning
- Unsupervised learning


# Quiz: Define the Problem
Which of the following problem statements fit the definition of a regression-based task?
    -detect when a cat jumps on a dinner table

    o determine the expected reading time for online news articles
    - determine if there are any collections of users that behave in similar ways
    o predict children shoe size for any particular age

"How can we increase the average number of minutes a customer spends listing on our app?"
    - This question is too broad with many different potential factors affecting how long a customer might spend listening to music.

    How ight you change the scope or redefine the questio to be better suited, and more concise, for a machine learning task?

    o Will changing the frequency of when we start playing ad affect how long a customer listens?
    o Will creating custom playlist encourage customers to listen longer?
    o Will creating artist interviews about their songs increase how long our customers spend listening?

## Build a Dataset
Understanding the data needed helps you select better models and algorithms so you can build more effective solutions.

## Data collection
Data collection can be as straightforward as running the appropriate SQL queries or as complicated as building customer web scraper applications to collect data for your project.
    - Does the data you've collected match the machine learning task and problem you have defined?

## Data inspection
The quality of your data will ultimately be the largest factor that affects how well you can expect your model to perform. As you inspect your data, look for:
    - Outliers
    - Missing or incomplete values
    - Data that needs to be transformed or preprocessed so it's in the correct format to be used by your model

## Summary statistics
Models can assume your data is structured.

Now that you have some data in hand it is a good best practice to check that your data is in line with the data underlying assumptions of your chosen machine learning model.

With many statistical tools, you can calculate things like the mean, inner-quartile range (IQR), and standard deviation. These tools can give you insight into the *scope, scale,* and *shape* of the dataset.

## Data visualization
You can use data visualization to see outliers and trends in your data to help stakeholders understand your data.


## How do we classify tasks when we don't have a label?
Unsupervised learning invloves using data that doesn't have a label. One common task is called clustering.
Clustering helps to determine if there are any naturally occurring groupings in the data.

In supervised learning, there are two main identifiers you will see in machine learning
    - A categorical label has a discrete set of possible values. 

    - A continuous (regression) label does not have a discrete set of possible values, which often means you are working with numerical data. 

## Model Training
What does a model training algorithm actually do?
    - Iteratively update model parameters to minimize some loss function

## Splitting your Dataset
The first step in model training is to randomly split the dataset. This allows you to keep some data hidden during training, so that data can be used to evaluate your model before you put it into production. Specifically, you do this to test against the bias-variance trade-off. If you're interested in learning more, see the Further learning and reading section.

Splitting your dataset gives you two sets of data:

    - Training dataset: The data on which the model will be trained. Most of your data will be here. Many developers estimate about 80%.
    - Test dataset: The data withheld from the model during training, whch is used to test how well your model will generalize to new data.

# Model Training Terminology
    The model training algorithm iteratively updates a model's parameters to minimize some loss function.

    - Model parameters: Model parameters are settings or configurations the training algorithm can update to change how the model behaves. Depending on the context, you'll also hear other more specific terms used to describe model parameters such as weights and biases. Weights, which are values that change as the learns, are more specific to neural networks.

    - Loss function: A loss function is used to codify the model's distance from its goal. For example, if you were trying to predict a number of snow cone sales based on the day's weather, you would care about making predictions that are as accurate as possible. So you might define a loss function to be "the average distance between your model's predicted number of snow cone sales and the correct number". You can see in the snow cone example this is the difference betweenb the two purple dots.



## Putting it All Together
The end-to-end training process is:
    - Feed the training data into the model
    - Compute the loss function on the results
    - Update the model parameters in the direction that reduces loss.

You continue to cycle through these steps until you reach a predefined stop condition. This might be based on a training time, the number of training cycles, or an even more intelligent or application-aware mechanism.


## Deep learning models
Extremely popular and powerful deep learning is a modern approach based around a conceptual model of how the human brain functions. The model is composed of collections of neurons connected together by weights. The process of training involves finding values for each weight.

Various neural network structures have been determined for modeling different kinds of problems or processing different kinds of data.

A short list of noteworthy examples includes:
    - FFNN: The most straightfoward way of structuring a neural network, the Feed Forward Neural Network structures neurons in a series of layers, with each neuron in a layer containing weights to all neurons in the previous layer.


    - CNN: Convolutional Neural Networks represent nested filters over grid-organized data. They are by far the most commonly used type of model when processing images.

    - RNN/LSTM: Recurrent Neural Networks and the related Long Short-Term Memory model types are structured to effectively represent for loops in traditional computing, collecting state while iterating over some object. They can be used for processing sequences of data.

    - Transformer: A more modern replacement for RNN/LSTMs, the transformer architecture enables training over larger datasets involving sequences of data.


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

- *Generative Adversarial Network (GAN)*: consist of two networks constantly competing with each other. 
    - Generator network that tries to generate data base on the data it was trained on.

    - Discriminator network that is trained to differentiate between real data and data which is created by the generator.

- *Binary classifier*: which means that the discriminator classifies inputs into two groups, "real" or "fake" data.


- *AR-CNN*: the autoregressive convolutional neural network generative technique uses a U-Net architecture. Makes iterative changes over time to create new data.

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

- *Naive Bayes*: is a machine learning algorithm used to solve classification problems. 

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



# Software Engineering practices Part I
In this lesson, you'll learn about the following software engineering practices and how they apply in data science.
    - Writing clean and odular code
    - Writing efficient code
    - Code refactoring
    - Adding meaningful documentation
    - Using version control

In the lesson following this one (part II), you'll also learn about the following software engineering practices:
    - Testing
    - Logging
    - Code reviews

## Clean and Modular Code
- *Productoin code*: Software running on production servers to handle live users and data of the intended audience. Note that this is different from *production-quality code*, which describes code that meets expectations for production in reliability, effeciency, and other aspects. Ideally, all code in production meets these expectations, but this is not always the case.

- *Clean code*: Code that is readable, simple, and concise. Clean production-quality code is crucial for collaboration and maintainability in software development.

- *Module code*: Code that is logically broken up into functions and modules. Modular production-quality code that makes your code more organized, efficient, and reusable.

- *Module*: A file. Modules allow code to be reused by encapsulating them into files that can be imported into other files.

## Quiz Question
- Which of the following describes code that is clean? Select all that apply.
    - Repetitive
    o Simple
    o Readable
    - Vague
    o Concise

- Making your code modular make it easier to do which of the following things? There is more than one correct answer.
    o Reuse your code
    o Write less code
    o Read your code
    o Collaborate on code

## Refactoring code
- *Refactoring*: Restructuring your code to improve its internal structure without changing its external functionality. This gives you a chance to clean and modularize your program after you've got it working

- Since it isn't easy to write your best code while you're still trying to just get it working, allocating time to do this is essential to producing high-quality code. Despite the initial time and effort required, this really pays off by speeding up your develoment time in the long run.

- You become a much stronger programmer when you're constantly looking to improve your code. The more you refactor, the easier it will be to structure and write good code the first time.


## Writing clean code: Meaningful names

Use meaningful names.
- Be *descriptive and imply type*: For booleans, you can prefix with `is_` or `has_` to make it clear it is a condition. You can also use parts of speech to imply types, like using verbs for functions and nouns for variables.

- Be *consistent but clearly differentiate*: `age_list` and `age` is easier to differentiate than `ages` and `age`. 

- *Avoid abbreviations and single letters*: You can determine when to make these exceptions based on the audience for your code. If you work with other data scientists, certain variables may be common knowledge. While if you work with full stack engineers, it might be necessary to provide more descriptive names in these cases as well. (Exceptions include counters and common math variables.)

- *Long names aren't the same as descriptive names*: You should be descriptive, but only with relevant information. For example, good function names describe what they do well without including details about implementation or highly specific uses.

Try testing how effective your names are by asking fellow programmer to guess the purpose of a function or variable based on its name, without looking at your code. Coming up with meaningful names often requires effort to get right.


## Writing clean code: Nice whitespace
Use whitespace properly.

- Organize your code with consistent indentation: the standard is to use four spaces for each indent.
    You can make this a default in your text editor.

- Separate sections with blank lines to keep your code well organized and readable.

- Try to limit your lines to around 79 characters, which is the guideline given in the PEP 8 style guide. In many good text editors, there is a setting to display a subtle line that indicates where the 79 character limit is.

For more guidelines, check out the code layout section of PEP 8 in the following notes.


pep8.md

## Quiz: Categorizing tasks
Quiz Question:
    Imagine you are writing a program that executes a number of tasks and categorizes each task based on its execution time. Below is a small snippet of this program.
    Which of the following naming changes could make this code cleaner? There may be more than one correct answer.

    ``` python
        t = end_time - start # compute execution time
        c = category(t) # get category of task
        print('Task Duration: {} seconds, Category: {}'.format(t, c))
    ```

    - None
    o Rename the variable `start` to `start_time` to make it consistent with `end_time`.
    o Rename the variable `t` to `execution_time` to make it more descriptive
    o Rename the function `category` to `categorize_task` to match the part of speech.
    o Rename the variable `c` to `category` to make it more descriptive
    

    ## Quize: Buying stocks
    Imagine you analyzed several stocks and calculated the ideal price, or limit price, at which you'd want to buy each stock.
    You write a program to iterate through your stocks and buy it *if the current price is below or equal to the limit price you computed*. Otherwise, you put it on a watchlist.
    Below are three ways of writing this code. Which of the following is the most clean?

```python

   # Choice A
stock_limit_prices = {'LUX': 62.48, 'AAPL': 127.67, 'NVDA': 161.24}
for stock_ticker, stock_limit_price in buy_prices.items():
    if stock_limit_price <= get_current_stock_price(ticker):
        buy_stock(ticker)
    else:
        watchlist_stock(ticker)
# Choice B
prices = {'LUX': 62.48, 'AAPL': 127.67, 'NVDA': 161.24}
for ticker, price in prices.items():
    if price <= current_price(ticker):
        buy(ticker)
    else:
        watchlist(ticker)
# Choice C
limit_prices = {'LUX': 62.48, 'AAPL': 127.67, 'NVDA': 161.24}
for ticker, limit in limit_prices.items():
    if limit <= get_current_price(ticker):
        buy(ticker)
    else:
        watchlist(ticker)
```

*Great job! All of the choices were passable, but Choice C was the most simple while also being descriptive. Choice A unnecessarily included the word stock everywhere, when we can already assume we are dealing with stocks based on the context. Naming everything with this can be redundant unless there is a clear reason to differentiate it with something similar. Choice B was also passable but could have more clearly differentiated the limit prices from the current price.*


## Writing module code
Follow the tips below to write modular code.

## Tip: DRY (Don't Repeat Yourself)
Don't repeat yourself! Modularization allows you to reuse parts of your code. Generalize and consolidate repeated code in functions or loops.

## Tip: Abstract out logic to improve readability
Abstracting out code into a function not only makes it less repetitive, but also improves readability with descriptive function names. Although your can become more readable when you abstract out logic into functions, it is possible to over-engineer this and have way too many modules, so use your judgement.

## Tip: Minimize the number of entities (functions, classes, modules, etc)
There are trade-offs to having function calls instead of inline logic. If you have broken up your code into an unnecessary amount of functions and modules, you'll have to jump around everywhere if you want to view the implementation details for something that may be too small to be worth it. Creating more modules doesn't necessarily result in effective modularization.

## Tip: Functions should do one thing
Each function you write should be focused on doing one thing. If a function is doing multiple things, it becomes more difficult to generalize and reuse. Generally, if there's an "and" in your function name, consider refactoring.

## Tip Arbitrary variable names can be more effective in certain functions
Arbitrary variable names in general functions can actually make the code more readable.

## Tip: Try to use fewer than three arguments per function
Try to use no more than three arguments when possible. This is not a hard rule and there times when it is more appropriate to use many parameters. But in many cases, it's more effective to use fewer arguments. Remember we are modularizing to simplify our code and make it more efficient. If your function has a log of parameters, you may want to rethink how you are splitting this up.

# Exercise: Refactoring - Wine quality
In this exercise, you'll refactor code that analyzes a wine quality dataset taken from the UCI Machine Learning Repository. Each row contains data on a wine sample, including several physicochemical properties gathered from tests, as well as a quality rating evaluated by wine experts.


```python
# solution code for wine_quality
import pandas as pd
df = pd.read_csv('winequality-red.csv', sep=';')
df.head()

## Renaming Columns

df.columns = [label.replace(' ', '_') for label in df.columns]
df.head()

## Analyzing Features

def numeric_to_buckets(df, column_name):
    median = df[column_name].median()
    for i, val in enumerate(df[column_name]):
        if val >= median:
            df.loc[i, column_name] = 'high'
        else:
            df.loc[i, column_name] = 'low' 

for feature in df.columns[:-1]:
    numeric_to_buckets(df, feature)
    print(df.groupby(feature).quality.mean(), '\n')
```

## Efficient code
Knowing how to write code that runs efficiently is another essential skill in software development.
Optimizing code to be more efficient can mean making it:
    - Execute faster
    - Take up less space in memory/storage

The project on which you're working determines which of these is more important to optimize for your company or product. When you're performing lots of different transformations on large amounts of data, this can make orders of magnitudes of difference in performance.

# Documentation
 - *Documentation*: Additional text or illustrated information that comes with or is embedded in the code of software.

 - Documentation is helpful for clarifying complex parts of code, making your code easier to navigate, and quickly conveying how and why different components of your program are used.

 - Several types of documentation can be added at different levels of your program:
    - *Inline comments* - line level
    - *Doctrings* - module and function level
    - *Project documentation* - project level

    ## Inline comments
    - Inline comments are text following hash symbols throughout your code. They are used to explain parts of your code, and really help future contributors understand your work

    - Comments often document the major steps of complex code. Readers may not have to understand the code to follow what it does if the comments explain it. However, others would argue that this is using comments to justify bad code, and that if code requires comments to follow, it is a sign refactoring is needed.

    - Comments are valuable for explaining where code cannot. For example, the history behind why a certain method was implemented a specific way. Sometimes an unconventional or seemingly arbitrary approach may be applied because of some obscure external variable causing side effects. These things are difficult to explain with code.


    ## Docstrings
    Docstring, or documentation strings, are valuable pieces of documentation that explain the functionality of any function or module in your code. Ideally, each of your functions should always have a docstring. 

    Docstrings are surrounded by triple quotes. The first line of the docstring is a brief explanation of the function's purpose.

    ### One-line docstring
    ```python
    def population_density(population, land_area):
    """Calculate the population density of an area."""
    return population / land_area
    ```

    If you think that the function is complicated enough to warrant a longer description, you can add a more thorough paragraph after the one-line summary.

    ### Multi-line docstring
    ```python
    def population_density(population, land_area):
    """Calculate the population density of an area.

    Args:
    population: int. The population of the area
    land_area: int or float. This function is unit-agnostic, if you pass in values in terms of square km or square miles the function will return a density in those units.

    Returns:
    population_density: population/land_area. The population density of a 
    particular area.
    """
    return population / land_area
    ```
    The next element of a docstring is an explanation of the function's arguments. Here, you list the arguments, state their purpose, and state what types the arguments should be. Finally, it is common to provide some description of the output of the function. Every piece of the docstring is optional; however,
    docstrings are part of good coding practice.

    ### Resources
    - PEP 257 - Docstring Conventions
    - Numpy Docstring Guide

    ### Project Documentation
    Project documentation is essential for getting others to understand why and how your code is relevant to them, whether they are potential users of your project or developers who may contribute to your code. A great first step in project documentation is your README file. It will often be the first interaction most users will have with your project.

    