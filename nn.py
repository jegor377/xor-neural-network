import math
from random import random, randint
import matplotlib.pyplot as plt


class NN:
	def __init__(self, lr=1.0):
		self.l1a1w1 = random()
		self.l1a1w2 = random()
		self.l1a2w1 = random()
		self.l1a2w2 = random()
		self.l1b1 = random()
		self.l1b2 = random()
		self.l1z1 = random()
		self.l1z2 = random()
		self.l1a1 = random()
		self.l1a2 = random()
		self.l2a1w1 = random()
		self.l2a1w2 = random()
		self.l2b1 = random()
		self.l2z1 = random()
		self.l2a1 = random()
		self.lr = lr

	def print(self):
		print(f"l1w11 = {self.l1a1w1}")
		print(f"l1w12 = {self.l1a1w2}")
		print(f"l1w21 = {self.l1a2w1}")
		print(f"l1w22 = {self.l1a2w2}")
		print(f"l1b1 = {self.l1b1}")
		print(f"l1b2 = {self.l1b2}")
		print('\n')
		print(f"l2w11 = {self.l2a1w1}")
		print(f"l2w12 = {self.l2a1w2}")
		print(f"l2b1 = {self.l2b1}")

	def feed_forward(self, i1, i2):
		self.l1z1 = self.l1a1w1 * i1 + self.l1a1w2 * i2 + self.l1b1
		self.l1a1 = activation(self.l1z1)
		
		self.l1z2 = self.l1a2w1 * i1 + self.l1a2w2 * i2 + self.l1b2
		self.l1a2 = activation(self.l1z2)

		self.l2z1 = self.l2a1w1 * self.l1a1 + self.l2a1w2 * self.l1a2 + self.l2b1
		self.l2a1 = activation(self.l2z1)

		return self.l2a1

	def calc_error(self, i1, i2, d):
		return (self.feed_forward(i1, i2) - d) ** 2.0

	def back_prop(self, i1, i2, d):
		error = self.calc_error(i1, i2, d)

		d_C_l2a1 = 2.0 * (self.l2a1 - d)

		d_l2a1_l2z1 = activation_der(self.l2z1)
		
		d_C_l2w11 = d_C_l2a1 * d_l2a1_l2z1 * self.l1a1
		d_C_l2w12 = d_C_l2a1 * d_l2a1_l2z1 * self.l1a2
		d_C_l2b1 = d_C_l2a1 * d_l2a1_l2z1

		d_C_l1a1 = d_C_l2a1 * d_l2a1_l2z1 * self.l2a1w1
		d_C_l1a2 = d_C_l2a1 * d_l2a1_l2z1 * self.l2a1w2

		d_l1a1_l1z1 = activation_der(self.l1z1)
		d_l1a2_l1z2 = activation_der(self.l1z2)

		d_C_l1w11 = d_C_l2a1 * d_l2a1_l2z1 * self.l2a1w1 * d_l1a1_l1z1 * i1
		d_C_l1w12 = d_C_l2a1 * d_l2a1_l2z1 * self.l2a1w1 * d_l1a1_l1z1 * i2
		d_C_l1w21 = d_C_l2a1 * d_l2a1_l2z1 * self.l2a1w2 * d_l1a2_l1z2 * i1
		d_C_l1w22 = d_C_l2a1 * d_l2a1_l2z1 * self.l2a1w2 * d_l1a2_l1z2 * i2

		d_C_l1b1 = d_C_l2a1 * d_l2a1_l2z1 * self.l2a1w1 * d_l1a1_l1z1
		d_C_l1b2 = d_C_l2a1 * d_l2a1_l2z1 * self.l2a1w2 * d_l1a2_l1z2

		self.l2a1w1 -= d_C_l2w11 * self.lr
		self.l2a1w2 -= d_C_l2w12 * self.lr

		self.l2b1 -= d_C_l2b1 * self.lr

		self.l1a1w1 -= d_C_l1w11 * self.lr
		self.l1a1w2 -= d_C_l1w12 * self.lr
		self.l1a2w1 -= d_C_l1w21 * self.lr
		self.l1a2w2 -= d_C_l1w22 * self.lr

		self.l1b1 -= d_C_l1b1 * self.lr
		self.l1b2 -= d_C_l1b2 * self.lr


def activation(x):
	return relu(x)

def activation_der(x):
	return relu_der(x)


def sigmoid(x):
	return 1.0 / (1.0 + math.e ** (-x))

def sigmoid_der(x):
	return sigmoid(x) * (1.0 - sigmoid(x))

def relu(x):
	if x > 0:
		return x
	return 0.01 * x

def relu_der(x):
	if x > 0:
		return 1.0
	return 0.01


def get_avg_error(brain):
	_errors = []
	for t in train_set:
		_errors.append(brain.calc_error(t[0], t[1], t[2]))
	return sum(_errors) / len(_errors)


train_set = [
	(0.0, 0.0, 0.0),
	(0.0, 1.0, 1.0),
	(1.0, 0.0, 1.0),
	(1.0, 1.0, 0.0)
]
errors = []


if __name__ == '__main__':
	learning_rate = 0.3
	brain = NN(learning_rate)
	for i in range(10_000):
		t_params = train_set[randint(0, 3)]
		brain.back_prop(t_params[0], t_params[1], t_params[2])
		errors.append(get_avg_error(brain))

	for t in train_set:
		print("SAMPLE: {} -> {} ({})".format(t, brain.feed_forward(t[0], t[1]), round(brain.feed_forward(t[0], t[1]))))
		print("ERROR: {}\n".format(brain.calc_error(t[0], t[1], t[2])))

	print('\n')
	brain.print()
	plt.plot(errors)
	plt.xlabel('Iterations')
	plt.title(f'(learning rate = {learning_rate})')
	plt.ylabel('Sample error per iteration')
	plt.show()