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

The nest step in machine learning process is to build a dataset that can be used to solve your machine learning-base problem. Understanding the data
needed helps you select better models and algorithms to build more effective solutions.

### The most important step of the machine learning process
Working with data is perhaps the most overlooked -- yet most important -- step of the machine learning process.

# The Four Aspects of Working with Data

## Data Collection
Data collection can be as straightforward as running the appropriate SQL queries or as complicated as building customer web scraper applications to collect data for your project.
    - Does the data you've collected match the machine learning task and problem you have defined?

## Data Inspection
The quality of your data will ultimately be the largest factor that affects how well you can expect your model to perform. As you inspect your data, look for:
    - Outliers
    - Missing or incomplete values
    - Data that needs to be transformed or preprocessed so it's in the correct format to be used by your model

## Summary Statistics
Models can assume your data is structured.

Now that you have some data in hand it is a good best practice to check that your data is in line with the data underlying assumptions of your chosen machine learning model.

With many statistical tools, you can calculate things like the mean, inner-quartile range (IQR), and standard deviation. These tools can give you insight into the *scope, scale,* and *shape* of the dataset.

## Data Visualization
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

##########################################[CaseStudy:Toolmarks]###################################################
# Case Study I
### Category: Classification (labeled data)
### ML Task: Supervised
### Model: 
### Data: (toolmark images, data reduction algorithms, product recommendations)
### Overview statement of machine learning problem
    This machine learning project wants to know if there is a relationship of product recommendations
    for tools that have a lesser degree of toolmark difference that other tools with toolmarks.

    Does it make sense to ask: "If this tool has a lesser degree of difference in toolmarks, will it get more recommendations?"
    Or maybe we can say: "Predict the number of recommendations for a tool"

    We feed our model this prediction
    We have labeled data

    We don't assume that because a tool as a specific or unique toolmarking that it will get more recommendations,
    rather we want to predict if the toolmark variation from tool to tool increases, will the recommendations start to decrees.

    The decrease in recommendations is not relative to the toolmark it-self
    The toolmark does not influence the recommendation it-self

    However,... It is our goal to see:
        "Given a tool has "greater" variations in its toolmarkings 
            - Can we predict the number of recommendations for that tool "

        "Given a tool has "fewer" variations in its toolmarkings
            - Can we predict the number of recommendations for that tool also "

Example:
    if a Hammer (in general sense) is produced that:
        1: has a toolmark (From manufacturing methods)
        2: has a 'low' variation of difference in this marking 'per tool'

    predict # or level of Recommendation (in general sense) for this product will be:
        1: high
        2: low
    
    So:
        Hammer: variation of difference = 0
        Recommendation: high

        vs

        Hammer: variation of difference = 1
        Recommendation: low

        this is does not assume because the variation is high that it will get low recommendations, only that we predict recommendations will not be high

        this also does not assume that because variation is low that it will get high recommendation, only that we predict recommendations will not be low

        this also does not assume that because recommendations are in fact high, variation is the cause in any sense,
        only that we predict high recommendations because we have samples of data to make projections with to conduct a study.

        this study does assume that data reduction algorithms influence the level of variation,
        and wager to one side (manufacturing method) more than the other due to level of acceptance in difference from the toolmark examiner community.



    Our graph data contains (degree of difference in toolmarks (for specific tool?)) and (the number of recommendations (for specific tool?))


### Analysis of Toolmarks
bagofworks = [toolsmarks, 
                toolmark examiner community, 
                statistical methods for data reduction and analysis (of images), 
                data reduction algorithms,
                rules used by examiners for classification and association of toolmarks, 
                critical toolmark match,
                product recommendation,
                manufacturing methods,
                statistical analysis,

                 ]

## Problem Statement:
    is the toolmark examiner community particularly accepting of manufacturing methods used to produce marks on tools that are substantially different from tool to tool? 

    or...

    at what level of difference between known matches and non-matches is acceptable in the toolmark examiner community? and does this influence product recommendations?

## Data Collection
        collection of digital images of toolmarks
        produced by various 'tool manufacturing methods' on 'produced work-products' -- 

        vs

        the development of 'statistical methods for data reduction and analysis of images'


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

    Whether it's an application or a package, your project should absolutely come with a README file. At a minimum, this should explain what it does, list its dependencies, and provide sufficiently detailed instructions on how to use it. Make it as simple as possible for others to understand the purpose of your project and quickly get something working.

    Translating all your ideas and thoughts formally on paper can be a little difficult, but you'll get better over time, and doing so makes a significant different in helping others realize the value of your project. Writing this documentation can also help you improve the design of your code, as you're forced to think through your design decisions more thoroughly. It also helps future contributors to follow your original intentions.

    There is a full Udacity course on this topic


# Version Control in Data Science
Git VCS and branching.


# Welcome to software engineering practices, part II
In part 2 of software engineering practices, you'll learn about the following practices of software engineering and how they apply in data science.
    - Testing
    - Logging
    - Code reviews

# Testing
    Testing your code is essential before deployment. It helps you catch errors and faulty conclusions before they make any major impact. Today, employers are looking for data scientists with skills to properly prepare their code for an industry setting, which includes testing their code.
    ## Testing and data science
        - Problems that could occur in data science aren't always easily detectable; you might have values being encoded incorrectly, features being used inappropriately, or unexpected data breaking assumptions.

        - To catch these errors, you have to check for the quality and accuracy of your *analysis* in addition to the quality of your *code*. Proper testing is necessary to avoid unexpected surprises and have confidence in your results.

        - Test-driven development(TDD): A development process in which you write tests for tasks before you even write the code to implement those tasks.

        - Unit test: A type of test that covers a "unit" of code -- usually a single function -- independently from the rest of the program.'

        # resources
        Four Ways Data Science Goes Wrong and How Test-Driven Data Analysis Can Help: Blog Post
        Ned Batchelder: Getting Started Testing: Slide Deck and Presentation Video


# Unit tests
We want to test our functions in a way that is repeatable and autoated. Ideally, we'd run a test progra that runs all our unit tests and cleanly lets us know which ones failed and which ones succeeded. Fortunately, there are great tools available in Python that we can use to create effective unit tests!

## Unit test advantages and disadvantages
The advantage of unit tests is that they are isolated from the rest of your program, and thus, no dependencies are involved. They don't require access to databases, APIs, or other external sources of information. However, passing unit tests isn’t always enough to prove that our program is working successfully. To show that all the parts of our program work with each other properly, communicating and transferring data between them correctly, we use integration tests. In this lesson, we'll focus on unit tests; however, when you start building larger programs, you will want to use integration tests as well.

To learn more about integration testing and how integration tests relate to unit tests, see Integration Testing. That article contains other very useful links as well

# Test-driven development and data science
- *Test-driven development*: Writing tests before you write the code that's being tested. Your test fails at first, and you know you've finished implementing a task when the test passes.

- Tests can check for different scenarios and edge cases before you even start to write your function. When start implementing your function, you can run the test to get immediate feedback on whether it works or not as you tweak your function.

- When refactoring or adding to your code, tests help you rest assured that your function behavior is repeatable, regardless of external paraeters such as hardware and time.

Test-driven development for data science is relatively new and is experiencing a lot of experimentation and breakthroughs. You can learn more about it by exploring the following resources.


# Logging
Logging is valuable for understanding the events that occur while running your program. For example, if you run your model overnight and the results the following morning are not what you expect, log messages can help you understand more about the context in those results occurred. Let's learn about qualities that make a log message effective.


# Log messages
Logging is the process of recording messages to describe events that have occurred while running your software. Let's take a look at a few examples, and learn tips for writing good log messages.

## Tip: Be professional and clear
```python
Bad: Hmmm... this isn't working???
Bad: idk.... :(
Good: Couldn't parse file.

```

## Tip: Be concise and use normal capitalization
```python
Bad: Start Product Recommendation Process
Bad: We have completed the steps necessary and will now proceed with the recommendation process for the records in our product database.
Good: Generating product recommendations.
```

## Tip: Choose the appropriate level for logging
Debug: Use this level for anything that happens in the program. Error: Use this level to record any error that occurs. Info: Use this level to record all actions that are user driven or system specific, such as regularly scheduled operations.

## Tip: Provide any useful information
Bad: Failed to read location data
Good: Failed to read location data: store_id 832



# Code reviews
Code reviews benefit everyone in a team to promote best programming practices and prepare code for production. Let's go over what to look for in a code review and some tips on how to conduct one.


## Questions to ask yourself when conducting a code review
First, let's look over some of the questions we might ask ourslves while reviewing code. These are drawn from the concepts we've covered in these last two lessons.

### Is the code clean and modular?
    - Can I understandthe code easily?
    - Does it use meaningful names and whitespace?
    - Is there duplicated code?
    - Can I provide another layer of abstraction?
    - Is each function and module necessary?
    - Is each function or module too long?

### Is the code efficient?
    - Are there loops or other steps I can vectorize?
    - Can I use better data structures to optimize any steps?
    - Can I shorten the number of calculations needed for any steps?
    - Can I use generators or multiprocessing to optimize any steps?

### Is the documentation effective?
    - Are inline comments concise and meaningful?
    - Is there complex code that's missing documentation?
    - Do functions use effective docstrings?
    - Is the necessary project documentation provided?

### Is the code well tested?
    - Does the code have high test coverage?
    - Do tests check for interesting cases?
    - Are the tests readable?
    - Can the tests be made more efficient?

### Is the logging effective?
    - Are log messages clear, concise, and professional?
    - Do they include all relevant and useful information?
    - Do they use the appropriate logging level?

## Tips for conducting a code review
Now that we know what we're looking for, let's go over some tips on how to actually write your code review. When your coworker finishes up some code that they want to merge to the team's code base, they might send it to you for review. You provide feedback and suggestions, and then they may make changes and send it back to you. When you are happy with the code, you approve it and it gets merged to the team's code base.

As you may have noticed, with code reviews you are now dealing with people, not just computers. So it's important to be thoughtful of their ideas and efforts. You are in a team and there will be differences in preferences. The goal of code review isn't to make all code follow your personal preferences, but to ensure it meets a standard of quality for the whole team.

### Tip: Use a code linter
This isn't really a tip for code review, but it can save you lots of time in a code review. Using a Python code linter like pylint can automatically check for coding standards and PEP 8 guidelines for you. It's also a good idea to agree on a style guide as a team to handle disagreements on code style, whether that's an existing style guide or one you create together incrementally as a team.

### Tip: Explain issues and make suggestions
Rather than commanding people to change their code a specific way because it's better, it will go a long way to explain to them the consequences of the current code and suggest changes to improve it. They will be much more receptive to your feedback if they understand your thought process and are accepting recommendations, rather than following commands. They also may have done it a certain way intentionally, and framing it as a suggestion promotes a constructive discussion, rather than opposition.

```
BAD: Make model evaluation code its own module - too repetitive.

BETTER: Make the model evaluation code its own module. This will simplify models.py to be less repetitive and focus primarily on building models.

GOOD: How about we consider making the model evaluation code its own module? This would simplify models.py to only include code for building models. Organizing these evaluations methods into separate functions would also allow us to reuse them with different models without repeating code.
```


### Keep your comments objective
Try to avoid using the words "I" and "you" in your comments. You want to avoid comments that sound personal to bring the attention of the review to the code and not to themselves.

```
BAD: I wouldn't groupby genre twice like you did here... Just compute it once and use that for your aggregations.

BAD: You create this groupby dataframe twice here. Just compute it once, save it as groupby_genre and then use that to get your average prices and views.

GOOD: Can we group by genre at the beginning of the function and then save that as a groupby object? We could then reference that object to get the average prices and views without computing groupby twice.
```

### Tip: Provide code examples
When providing a code review, you can save the author time and make it easy for them to act on your feedback by writing out your code suggestions. This shows you are willing to spend some extra time to review their code and help them out. It can also just be much quicker for you to demonstrate concepts through code rather than explanations.

Let's say you were reviewing code that included the following lines:

```python
first_names = []
last_names = []

for name in enumerate(df.name):
    first, last = name.split(' ')
    first_names.append(first)
    last_names.append(last)

df['first_name'] = first_names
df['last_names'] = last_names
```

```
BAD: You can do this all in one step by using the pandas str.split method.
GOOD: We can actually simplify this step to the line below using the pandas str.split method. Found this on this stack overflow post: https://stackoverflow.com/questions/14745022/how-to-split-a-column-into-two-columns
```

```python
df['first_name'], df['last_name'] = df['name'].str.split(' ', 1).str

```


# Introduction to Object-Oriented Programming

## Lesson outline
    - Object-oriented programming syntax
        - Procedural vs. object-oriented programming
        - Classes, objects, methods and attributes
        - Coding a class
        - Magic methods
        - Inheritance

    - Using object-oriented programming to make a Python package
        - Making a package
        - Tour of scikit-learn source code
        - Putting your package on PyPi

## Why object-oriented programming?
Object-oriented programming has a few benefits over procedural programming, which is the programming style you most likely first learned. As you'll see in this lesson:
    - Object-oriented programming allows you to create large, modular programs that can easily expand over time.
    - Object-oriented programs hide the implementation from the end user.


Consider Python packages like Scikit-learn, pandas, and NumPy. These are all Python packages built with object-oriented programming. Scikit-learn, for example, is a relatively large and complex package built with object-oriented programming. This package has expanded over the years with new functionality and new algorithms.

When you train a machine learning algorithm with Scikit-learn, you don't have to know anything about how the algorithms work or how they were coded. You can focus directly on the modeling.

Here's an example taken from the Scikit-learn website:

```python
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y) 
```

How does Scikit-learn train the SVM model? You don't need to know because the implementation is hidden with object-oriented programming. If the implementation changes, you (as a user of Scikit-learn) might not ever find out. Whether or not you should understand how SVM works is a different question.

In this lesson, you'll practice the fundamentals of object-oriented programming. By the end of the lesson, you'll have built a Python package using object-oriented programming.


### Lesson files
This lesson uses classroom workspaces that contain all of the files and functionality you need. You can also find the files in the data scientist nanodegree term 2 GitHub repo.


