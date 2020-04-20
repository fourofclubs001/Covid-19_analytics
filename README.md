# Covid-19_analytics

Approximate covid-19 new infected or deaths curve per day through existing data

## demo
![Canada cases approximate curve][logo]

[logo]: https://github.com/fourofclubs001/Covid-19_analytics/blob/master/Spain_cases_curve.png "Spain cases approximate curve"
## set up

### requierement
requiered: 
- python 3.6.8
- matplotlib 3.1.1
- tensorflow 2.0.0
- tqdm 4.36.1
- numpy 1.17.2
- csv 1.0

### data set
The data set being use is from the "European Centre for Disease Prevention and Control" (An agency of the European Union) and it can be download from the following [link](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide)

### Try it
1. Clone this repository
2. Run the program
'''
python covid-19_analytics.py
'''
3. Insert desire country, cases or deaths data to analize, and quantity of epochs to train the model. Ex:
'''
country: Canada
cases or deaths: cases
epochs to train: 1000
'''

## Abstract
In terms of epidemic or pandemic situation, it is important for every government level(city, state and country) to make decisions based on how the situation will evolve.

This program objective is to make trustable predictions based on the pandemic available data and the normal pandemic behaviours

## Algorithm
The algorithm is divided on three parts:

### Data processing
On first place the data is collected from the 'data.csv' file. The program save the desired country cases or death on a single variable call 'data_set'. Then the data is normalized for a correct data analysis

### Data analysis
After that a function is 'train' using the normalized data. This function is the sigmoid function derivative and it can be adjust to any epidemic data through 3 parameters which I will call 'a', 'd', and 'h'. Being f(x) the sigmoid function derivative, curve_tf(a, d, h, x) = a.f(d-h.x) will be the parameterized function.
In general terms 'a' will adjust the amplitud, 'd' the distance of the peak from the origin, and 'h' the curve width.

The function 'training' consist on approximate the existing data points to the curve. For doing so, on first place the peak of the untrained function is placed above the highest data point, for making easy the curve approximation. On the second place 'training process' take place. This process is 'gradient descent', where the MSE (mean square error) is calculated between each data point and it correspoding point on the x value of the function, and then the 'a', 'd' and 'h' parameters are optimized. The 'gradient descent' process is repeated as many 'epochs' as the user wants.

### Data visualization
Finally the data set and the trained function are visualized on plot. In addition, some critic points (fist and last case or death stimated point, function inflexion points and the peak point) are calculated and visualized too.

## Question to author
For any question email to: lucasvitali001@gmail.com