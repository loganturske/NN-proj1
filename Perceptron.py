import math

class Perceptron:


	def __init__(self, weights, bias, desired_output, eta):
		self.weights = weights
		self.bias = bias
		self.desired_output = desired_output
		self.inputs = [0,0]
		self.eta = eta
		self.activity = None
		self.activation = None
		self.delta = None
		self.new_bias = bias
		self.weighted_sum = 0
		self.old_weights = weights

	def print(self):
		print("Weights   : " + str(self.weights))
		print("Activity  : " + str(self.activity))
		print("Activation: " + str(self.activation))
		print("Delta     : " + str(self.delta))
		print("Eta 		 : " + str(self.eta))
		print("Inputs    : " + str(self.inputs))
		print("Bias      : " + str(self.bias))
		print("Old       : " + str(self.old_weights))

	def calc_activity(self):

		# Set a running total
		running_total = 0
		# Sum all the weights*inputs
		for index in range(len(self.inputs)):
			running_total += self.inputs[index] * self.weights[index]

		self.activity = running_total + self.bias

	def calc_activation(self, activity_value):
		# Sigmoidal
		self.activation = 1/(1+math.exp(-1*activity_value))

	def set_delta_weights(self):
		for index in range(len(self.weights)):
			self.weights[index] = self.weights[index] + (self.eta * self.delta * self.inputs[index])
		self.bias = self.bias + (self.eta*self.delta)
		
	def update_weights(self):
		for index in range(len(self.weights)):
			self.weights[index] = self.weights[index] + (self.eta * self.delta * self.inputs[index])
		self.bias = self.bias + (self.eta*self.delta)

	def get_activation(self):
		return self.activation

	def save_old_weights(self):
		self.old_weights = self.weights[:]

	def calc_delta(self):
		self.delta = (self.desired_output - self.activation)*(1 - self.activation)*self.activation

	def set_inputs(self, inputs):
		for index in range(len(inputs)):
			self.inputs[index] = inputs[index]

	def set_desired_output(self, desired_output):
		self.desired_output = desired_output

	def set_delta(self, delta):
		self.delta = delta
