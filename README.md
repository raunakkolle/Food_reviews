# Food_reviews


 ABSTRACT

Sentiment analysis deals with the identification and classification of opinions and sentiments expressed on text.  Interpersonal interactions have become a customary standard in the modern era. With the easy access to the internet, more and more people have now become opinionated. Appreciation as well as criticism often have good numbers amongst the masses. These opinions are often left behind as reviews of services and goods. Words speak the perspectives of the people and with the help of the electronic systems administration media, the internet, people are able to post their perspectives anonymously without any kind of hesitation.The classification of these reviews as positive, negative and nonpartisan polarities can become a formidable task in the assessment of NLP. These reviews and feedback from consumers tends to be more subjective than factual. The sentiment analysis of customer reviews can aid a company to overcome it’s lacking and therefore guide them to achieve better improvements. This project performs a supervised learning task to predict the score of  amazon fine food reviews on a scale of 1-5 using logistic regression and classifying them under the categories of positive, negative and nonpartisan polarities through multi-class text classification.

       Keywords: reviews, polarities, logistic regression, supervised learning, multi-class text    classification.

INTRODUCTION

Sentiment analysis is the technique of mechanically analysing a large number of product evaluations and extracting relevant and meaningful information in order to determine whether or not customers are truly happy with the product. The feedback of a person is more subjective than objective. Negative, positive, or neutral feedback is possible. Natural Language Processing (NLP) and Text Analysis methods are used in sentiment analysis to emphasise the subjective information in a text. Using sentiment analysis to analyse client evaluations can help you figure out what's missing and, as a result, point you in the right direction for development.The negative remarks and reasons why customers have difficulties with your product or service can be identified by reviewing customer feedback and answers. The suggested approach employs Logistic Regression and Supervised Learning Multi-Class Text Classification. The family of classifiers known as exponential or log-linear classifiers includes logistic regression. Its log-linear classifier, like Naive Bayes, operates by extracting a collection of weighted features from the input, taking logs, and linearly merging them (meaning that each feature is multiplied by a weight and then added up). A classifier that classifies an observation into one of two groups is known as logistic regression, while multinomial logistic regression is used when there are more than two classes. The key goal is to figure out how a reviewer feels about it. (Negative/Neutral/Favorable) Our research might be beneficial in helping restaurants better comprehend what reviews are saying about their cuisine, as well as other jobs like recommender systems.

 
Types of Logistic Regression
Logistic Regression models can be classified into three groups based on the target variable categories. These three groups are described below:-
Binary Logistic Regression
In Binary Logistic Regression, the target variable has two possible categories. The common examples of categories are yes or no, good or bad, true or false, spam or no spam and pass or fail.
Multinomial Logistic Regression
In Multinomial Logistic Regression, the target variable has three or more categories which are not in any particular order. So, there are three or more nominal categories. The examples include the type of categories of fruits - apple, mango, orange and banana.
Ordinal Logistic Regression
In Ordinal Logistic Regression, the target variable has three or more ordinal categories. So, there is intrinsic order involved with the categories. For example, the student performance can be categorized as poor, average, good and excellent.



DATASET

The Amazon Fine Food Reviews dataset, which comprises 568,454 reviews.The data is included in a single CSV file that contains the product ids, reviewer ids, reviewer scores (ranging from 1 to 5), the date for each review, a brief synopsis for each review, and the text of the reviews. As labels and raw inputs, we extract the columns of scores and review texts. The following information is included in the data:

- 568,454 reviews
- 256,059 users
- 74,258 products

Attributes :
- ProductId : Unique identifier for the product
- UserId : Unique identifier for the user
- ProfileName : Name of customer
- HelpfulnessNumerator : Number of users who found the review helpful
- HelpfulnessDenominator : Number of users who indicated whether they found the review helpful or not
- Score : Rating between 1 and 5
- Time : Timestamp for the review
- Summary : Brief summary of the review
- Text : Review text




METHODOLOGY
 
Logistic Regression intuition
In statistics, the Logistic Regression model is a widely used statistical model which is primarily used for classification purposes. It means that given a set of observations, the Logistic Regression algorithm helps us to classify these observations into two or more discrete classes. So, the target variable is discrete in nature.
The Logistic Regression algorithm works as follows:-
Implement linear equation
Logistic Regression algorithm works by implementing a linear equation with independent or explanatory variables to predict a response value. For example, we consider the number of hours studied and probability of passing the exam. Here, the number of hours studied is the explanatory variable and it is denoted by x1. Probability of passing the exam is the response or target variable and it is denoted by z.
If we have one explanatory variable (x1) and one response variable (z), then the linear equation would be given mathematically with the following equation-
z = β0 + β1x1
 
Here, the coefficients β0 and β1 are the parameters of the model.
If there are multiple explanatory variables, then the above equation can be extended to
z = β0 + β1x1+ β2x2+……..+ βnxn
 
Here, the coefficients β0, β1, β2 and βn are the parameters of the model.
So, the predicted response value is given by the above equations and is denoted by z.
Sigmoid Function
This predicted response value, denoted by z is then converted into a probability value that lies between 0 and 1. We use the sigmoid function in order to map predicted values to probability values. This sigmoid function then maps any real value into a probability value between 0 and 1.
In machine learning, sigmoid function is used to map predictions to probabilities. The sigmoid function has an S shaped curve. It is also called sigmoid curve.
A Sigmoid function is a special case of the Logistic function. It is given by the following mathematical formula.
Graphically, we can represent the sigmoid function with the following graph.

Decision boundary
The sigmoid function returns a probability value between 0 and 1. This probability value is then mapped to a discrete class which is either “0” or “1”. In order to map this probability value to a discrete class (pass/fail, yes/no, true/false), we select a threshold value. This threshold value is called the Decision boundary. Above this threshold value, we will map the probability values into class 1 and below which we will map values into class 0.
Mathematically, it can be expressed as follows:-
p ≥ 0.5 => class = 1


p < 0.5 => class = 0 
 
Generally, the decision boundary is set to 0.5. So, if the probability value is 0.8 (> 0.5), we will map this observation to class 1. Similarly, if the probability value is 0.2 (< 0.5), we will map this observation to class 0.

Making predictions
Now, we know about sigmoid function and decision boundary in logistic regression. We can use our knowledge of sigmoid function and decision boundary to write a prediction function. A prediction function in logistic regression returns the probability of the observation being positive, Yes or True. We call this class 1 and it is denoted by P(class = 1). If the probability inches closer to one, then we will be more confident about our model that the observation is in class 1.
In the previous example, suppose the sigmoid function returns the probability value of 0.4. It means that there is only a 40% chance of passing the exam. If the decision boundary is 0.5, then we predict this observation as a failure.
Cost function
In this case, the prediction function is nonlinear due to the sigmoid transformation. We square this prediction function to get the mean square error (MSE). It results in a non-convex function with many local minimums. If the cost function has many local minimums, then the gradient descent may not converge and do not find the global optimal minimum. So, instead of mean square error (MSE), we use a cost-function called Cross-Entropy.
Cross-Entropy
Cross-Entropy is a cost-function which measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-Entropy loss is also known as Log Loss. It can be divided into two separate cost-functions: one for y = 1 and one for y = 0.
Mathematically, it can be given with the following formula.

The cross-entropy loss function can be represented with the following graphs for y = 1 and y = 0. These are smooth monotonic functions which always increase or always decrease. They help us to easily calculate the gradient and minimize cost.

Cross-entropy loss increases as the predicted probability diverges from the actual label. This cost-function penalizes confident and wrong predictions more than it rewards confident and right predictions. A perfect model would have a log loss of zero.
The above loss-functions can be compressed into one function as follows.

In binary classification models, where the number of classes is equal to 2, cross-entropy can be calculated as follows.
	-(y log (p) + (1 – y) log (1 – p))
 
If there is a multiclass classification, we calculate a separate loss for each class label per observation and sum the result as follows.
	-Ʃ yo,c log (po,c) 
 
Here,
Summation is over the number of classes.
log – natural logarithm
y – binary indicator (0 or 1) if class label c is correctly classified for observation o.
predicted probability observation o is of class C.
Vectorized cost function can be given as follows.

Gradient descent
To minimize the cost-function, we use gradient descent technique. Python machine-learning library Scikit-learn hide this implementation.
The derivative of the sigmoid function is given by the following formula.
	sʹ(z) = s(z) (1 - s(z))
 
The above equation leads us to the cost-function given by the following formula.
	Cʹ = x (s(z) - y)
 
Here, we have
C′ is the derivative of cost with respect to weights
y is the actual class label (0 or 1)
s(z) is the model prediction
x is the feature vector
Mapping probabilities to classes
The final step is to assign class labels (0 or 1) to the predicted probabilities.
 Assumptions of Logistic Regression
Logistic Regression does not require the key assumptions of linear regression and generalized linear models. In particular, it does not require the following key assumptions of linear regression:-
Logistic Regression does not follow the assumption of linearity. It does not require a linear relationship between the independent and dependent variables.
The residuals or error terms do not need to follow the normal distribution.
Logistic Regression does not require the assumption of homoscedasticity. Homoscedasticity means all the variables in the model have the same variance. So, in the Logistic Regression model, the variables may have different variance.
The dependent variable in Logistic Regression is not measured on an interval or ratio scale.
The Logistic Regression model requires several key assumptions. These are as follows:-
Logistic Regression model requires the dependent variable to be binary, multinomial or ordinal in nature.
It requires the observations to be independent of each other. So, the observations should not come from repeated measurements.
Logistic Regression algorithm requires little or no multicollinearity among the independent variables. It means that the independent variables should not be too highly correlated with each other.
The Logistic Regression model assumes linearity of independent variables and log odds.
The success of the Logistic Regression model depends on the sample sizes. Typically, it requires a large sample size to achieve the high accuracy.

RESULTS AND CONCLUSION

The logistic regression model accuracy score is 0.8501. So, the model does a very good job in predicting whether or not it will rain tomorrow in Australia.
Small number of observations predict that there will be rain tomorrow. Majority of observations predict that there will be no rain tomorrow.
The model shows no signs of overfitting.
Increasing the value of C results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.
Increasing the threshold level results in increased accuracy.
ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.
Our original model accuracy score is 0.8501 whereas accuracy score after RFECV is 0.8500. So, we can obtain approximately similar accuracy but with a reduced set of features.
In the original model, we have FP = 1175 whereas FP1 = 1174. So, we get approximately the same number of false positives. Also, FN = 3087 whereas FN1 = 3091. So, we get slightly higher false negatives.
Our original model score is found to be 0.8476. The average cross-validation score is 0.8474. So, we can conclude that cross-validation does not result in performance improvement.
Our original model test accuracy is 0.8501 while GridSearchCV accuracy is 0.8507. We can see that GridSearchCV improves the performance for this particular model.
