# Fraud Model Counterfactual Evaluation Simulator

## Goal

The goal of the this project is to generate a simulator to test multiple scenarios of Counterfactual Evaluation of a syntetic fraud model. The counterfactual assessment occurs, because a share of the transaction that would be blocked by the model are allowed at random, and this data is used to estimate what would would be the model performance if all transactions were allowed. The parameter we want to variate constantly is the pdf of the model scores, the distribution of fraud in relation to the fraud model values, the percentage of blocked transaction allowed at random; and how all of these impact the estimations (mostly the variance of its estimator).

## Features

- Data Generator:
    - The input is the beta distribution alpha and beta parameters, normal mean and sd, and the sample size
    - the default values for the parameters are: alpha=0.5, beta=10, mean=0, sd=0.1, sample_size=10_000
    - generate the model scores array based on a beta.rvs(a=alpha, b=beta, size=sample_size)
    - generate an array of the size of model scores from np.random.normal(mean, sd, sample_size) as model error
    - sum model error and model socre, clipe their values to be between 0 and 1 and, lastly, sample from binomial to generate the fraud array of 0 and 1 
    - return a dataframe with all arrays
- Logging Policy Generator:
    - It should simulate fraud prevention system policy (aka which orders would be blocked and not) based on the cutoff value (the value for which the model would flag a model score as fraudulent), and the exploration rate (percentage of the blocked transactions that will be allowed at random)
    - The input is the data generator output, the mode cutoff (float with 0.05 as default value),and the exploration rate (float with 0.5 as default value)
    - The function will add a propensity score column (ps) that will have value 1 for allowed transactions, 0.95 for transactions bloked, and 0.05 for transaction selected at random from the blocked transaction to be allowed
    - The function will add a action column containing the final action executed for the row
- Counterfactual Values Estimator
    - it should apply Counterfactual concepts to the Logging Policy Generator dataset to calculate policy metrics mean and CI
    - The input is the Logging Policy Generator returned dataframe as input, and the number of repetitions for the bootstrap estimator, and which metrics to calculate (default are precision and recall, but it should be flexible to accept a list of custom metrics)
    - it should create a column for weights = 1 / ps score
    - it should filter out blocked transaction from the calculation
    - it should calculate a poisson array for each row (np.random.poisson(1, (n_rows, n_simulations)))
    - it should generate a final vector for each row based on multipling each row element by the fraud label and the weight column
    - it should calculate the metrics for each element of the array position using all rows
    - it should calculate the mean and the percentile 2.5 and 97.5 percentile from this final array. This will be the mean and 95% Confidence Interval for the metric
    - return a dictonary containing the metrics along with its mean, p250 and p975
- Off Policy Evaluation Pipeline
    - it should take all relevante distribution parameters and the exploration rate as input and execute the whole pipeline above
- Off Policy Evaluation Simulator
    - It should take the Off Policy Evaluation Pipeline and iterate over multiple exploration rate values (eg np.linspace(0.01, 0.1, 10)). By the end of it, it should plot the model score pdf, a calibration plot and a precision recall plot based on the model scores and fraud values, and, lastly, a plot for each metric along with how each value (and its CI) changed according to the amount of exploration
