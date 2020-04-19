import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import math
import csv

country = 'Canada' # Canada, Italy, Spain
data = 'cases' # cases or deaths

epochs = 10
lr = 1

# Read data from csv

with open('data.csv', 'r', encoding = 'utf-8') as csv_file:

	file_reader = csv.DictReader(csv_file)

	data_rate = []

	for row in file_reader:

		if row['countriesAndTerritories'] == country:

			data_rate.append(int(row[data]))



# Data normalization

def normalize(data_set):

	# Normalize data_set function
	# set days and data numbers between 0 and 1
	# return nomalize data_set and maximum values for denormalize

	max_values = np.amax(data_set, axis = 0)

	for i in range(len(data_set)):

		data_set[i] = [data_set[i][0]/max_values[0], data_set[i][1]/max_values[1]]

	return data_set, max_values

# Organize data
data_set = []

for i in range(len(data_rate)):

	data_set.append([len(data_rate) - i - 1, data_rate[i]])

# Normalize data_set
data_set, max_values = normalize(data_set)



# Data curve aproximation

# Curve function: 
# a: amplitude
# d: x displacement
# h: arc width
# x: indepent variable

def curve_tf(a, d, h, x):
	
	# Curve function

	num = tf.math.multiply(tf.cast(a, 'float32'), tf.math.pow(tf.Variable(math.e), tf.cast(tf.math.subtract(d, tf.math.multiply(h, x)), 'float32')))
	den = tf.math.pow(tf.math.add(1, tf.math.pow(tf.Variable(math.e), tf.cast(tf.math.subtract(d, tf.math.multiply(h, x)), 'float32'))),2)
	res = tf.math.divide(num, den)

	return res

def calculate_mse_tf(data_set, a, d, h):

	# MSE (mean square error) function

	res = tf.Variable(0, dtype = 'float32')
	
	for i in range(len(data_set)):

		res = tf.math.add(res, tf.math.pow(tf.math.subtract(curve_tf(a, d, h, data_set[i][0]), data_set[i][1]), 2))

	res = tf.math.divide(res, len(data_set))

	return res

# Learning loop

# Data initialization

a = 1
h = 20
# d such that the curve peek is below the highest data point
d = data_set[np.argmax(data_set, axis = 0)[1]][0]*h

a = tf.Variable(a, dtype = 'float32')
d = tf.Variable(d, dtype = 'float32')
h = tf.Variable(h, dtype = 'float32')

error_history = [] 

for e in tqdm(range(epochs)):

	with tf.GradientTape() as t:

		t.watch(a)
		mse = calculate_mse_tf(data_set, a, d, h)
		a_derivative = t.gradient(mse, a)

	with tf.GradientTape() as t:

		t.watch(d)
		mse = calculate_mse_tf(data_set, a, d, h)
		d_derivative = t.gradient(mse, d)

	with tf.GradientTape() as t:

		t.watch(h)
		mse = calculate_mse_tf(data_set, a, d, h)
		h_derivative = t.gradient(mse, h)

	error_history.append(mse.numpy())

	a = tf.math.subtract(a, lr*a_derivative)
	d = tf.math.subtract(d, lr*d_derivative)
	h = tf.math.subtract(h, lr*h_derivative)

a = a.numpy().astype(np.float64)
d = d.numpy().astype(np.float64)
h = h.numpy().astype(np.float64)
mse = mse.numpy().astype(np.float64)

# Denormilze data and curve

def denormalize_data(data_set, max_values):

	for i in range(len(data_set)):

		data_set[i] = [data_set[i][0]*max_values[0], data_set[i][1]*max_values[1]]

	return data_set

# Denormalize data
data_set = denormalize_data(data_set, max_values)

# Denormalize curve parameters
h = h/max_values[0]
a = a*max_values[1]

# Calculate critic points 

critic_p_x = [(d-np.log((a-2+math.sqrt(math.pow(a, 2)-(4*a)))/2))/h,
              (d-np.log(2+math.sqrt(3)))/h,
              d/h,
              (d-np.log(2-math.sqrt(3)))/h,
              (d-np.log((a-2-math.sqrt(math.pow(a, 2)-(4*a)))/2))/h]

critic_p_y = []

for point in critic_p_x:

	critic_p_y.append(curve_tf(a, d, h, point).numpy())

# Plot

# Data organization
data_x = []
data_y = []

for i in range(len(data_set)):

	data_x.append(data_set[i][0])
	data_y.append(data_set[i][1])

curve_x = np.arange(0, 2*max_values[0], 0.1)
curve_y = []

for i in curve_x:

	curve_y.append(curve_tf(a, d, h, i).numpy())

# data and curve
plt.plot(curve_x, curve_y)
plt.scatter(data_x, data_y, c = 'orange', s = max_values[1]/400)

# critic points
plt.scatter(critic_p_x, critic_p_y, c = 'green')

for point in critic_p_x:

	plt.plot([point,point], [-max_values[1]/49, max_values[1] + max_values[1]/49], c = [0,0,0,0.1])

for point in critic_p_y:

	plt.plot([critic_p_x[0] - critic_p_x[0]/24, critic_p_x[-1] + critic_p_x[-1]/24], [point,point], c = [0,0,0,0.1])

# plot limits
plt.xlim(critic_p_x[0] - critic_p_x[0]/25, critic_p_x[-1] + critic_p_x[-1]/25)
plt.ylim(-max_values[1]/50, max_values[1] + max_values[1]/50)

# plot text
maxlimy = max_values[1] + max_values[1]/50
line_space = maxlimy/10 

# critic points texts
plt.text(critic_p_x[0] + critic_p_x[0]/25, maxlimy - line_space, f'peak: {int(critic_p_x[2])}')
plt.text(critic_p_x[0] + critic_p_x[0]/25, maxlimy - 2*line_space, f'critic point: {int(critic_p_x[1])}')
plt.text(critic_p_x[0] + critic_p_x[0]/25, maxlimy - 3*line_space, f'critic point: {int(critic_p_x[3])}')
plt.text(critic_p_x[0] + critic_p_x[0]/25, maxlimy - 4*line_space, f'first {data[:-1]}: {int(critic_p_x[0])}')
plt.text(critic_p_x[0] + critic_p_x[0]/25, maxlimy - 5*line_space, f'last {data[:-1]}: {int(critic_p_x[4])}')

# parameters text
plt.text(critic_p_x[4] - 8*(critic_p_x[0]/25), maxlimy - line_space, f'a: {round(a, 2)}')
plt.text(critic_p_x[4] - 8*(critic_p_x[0]/25), maxlimy - 2*line_space, f'd: {round(d, 2)}')
plt.text(critic_p_x[4] - 8*(critic_p_x[0]/25), maxlimy - 3*line_space, f'h: {round(h, 2)}')
plt.text(critic_p_x[4] - 8*(critic_p_x[0]/25), maxlimy - 4*line_space, f'mse: {round(mse, 2)}')

# title and labels
plt.title('Curve approximation from data')
plt.ylabel(f'new {data} per day')
plt.xlabel('days')

# show plot
plt.show()