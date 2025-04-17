## logistic regression
- Logistic regression is a statistical method for predicting binary classes. The outcome is usually coded as 0 or 1, where 1 indicates the presence of a characteristic or event and 0 indicates its absence. Logistic regression estimates the probability that a given input point belongs to a certain class. It uses the logistic function to model the relationship between the input features and the binary outcome.
- Logistic regression is a type of regression analysis used for prediction of outcome of a categorical dependent variable based on one or more predictor variables. The outcome is usually binary (0/1, yes/no, true/false). It is a special case of the generalized linear model (GLM) and is used when the dependent variable is binary or categorical.

### loss function 
- The loss function for logistic regression is the binary cross-entropy loss, also known as log loss. It measures the performance of a classification model whose output is a probability value between 0 and 1. The binary cross-entropy loss is defined as:
$$
L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$
where:
- \(L\) is the loss function
- \(y\) is the true label (0 or 1)
- \(\hat{y}\) is the predicted probability of the positive class (1)
- \(N\) is the number of samples
- \(y_i\) is the true label for the \(i\)-th sample
- \(\hat{y}_i\) is the predicted probability for the \(i\)-th sample
- The loss function quantifies the difference between the predicted probabilities and the actual labels. The goal of logistic regression is to minimize this loss function during training. The binary cross-entropy loss is particularly useful for binary classification problems, as it penalizes incorrect predictions more heavily than correct ones.


### cost function 
- The cost function for logistic regression is the same as the loss function, which is the binary cross-entropy loss. The cost function measures how well the model's predictions match the true labels. The goal of logistic regression is to minimize this cost function during training. The cost function is defined as:
$$
J(\theta) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(h_\theta(x_i)) + (1 - y_i) \log(1 - h_\theta(x_i))]
$$
where:
- \(J(\theta)\) is the cost function
- \(h_\theta(x_i)\) is the predicted probability of the positive class (1) for input \(x_i\)
- \(\theta\) is the vector of model parameters (weights)
- The cost function is minimized using optimization algorithms such as gradient descent or stochastic gradient descent (SGD). The optimization process adjusts the model parameters to find the best fit for the training data, ultimately leading to better predictions on unseen data.