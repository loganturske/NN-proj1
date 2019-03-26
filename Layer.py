from Perceptron import Perceptron
class Layer:


	def __init__(self, output_layer_bool, num_of_inputs, num_of_perceptrons, weights, biases, desired_output, eta):
		self.outputLayerFlag = output_layer_bool
		self.layer = []
		self.layer_len = num_of_perceptrons
		self.littleE_vector = None
		self.desired_output = desired_output
		for index in range(num_of_perceptrons):
			# print(num_of_perceptrons)
			self.layer.append(Perceptron(weights[index], biases[index], desired_output, eta))

	def print(self):
		for index in range(len(self.layer)):
			self.layer[index].print()

	def print_error(self):
		print(self.littleE_vector)

	def print_big_E(self):
		e = .5*(self.littleE_vector*self.littleE_vector)
		print(e)

	def get_big_E(self):
		return .5*(self.littleE_vector*self.littleE_vector)

	def set_desired_output(self, desired_output):
		self.desired_output = desired_output
		for index in range(len(self.layer)):
			self.layer[index].set_desired_output(desired_output)

	def get_error_vector(self):
		self.littleE_vector = self.desired_output - self.layer[0].activation
		return self.littleE_vector

	def get_layer_output_vector(self):
		vec = []
		for index in range(len(self.layer)):
			self.layer[index].calc_activity()
			self.layer[index].calc_activation(self.layer[index].activity)
			vec.append(self.layer[index].activation)
		return vec

	def get_layer_length(self):
		return self.layer_length

	def set_output_layer_delta_values():
		for index in range(len(layer)):
			layer[index].set_delta_weights()

	def set_hidden_layer_delta_values(self, hidden_layer):
		hidden_layer = hidden_layer.layer

		for output_node in self.layer:
			index = 0
			for hidden_node in hidden_layer:
				d = (1-hidden_node.activation)*(hidden_node.activation)*(output_node.delta * output_node.old_weights[index])
				hidden_node.set_delta(d)
				index = index + 1

	def calc_layer_delta_weights(self):
		for node in self.layer:
			node.save_old_weights()
			node.calc_delta()
			node.set_delta_weights()

	def udpate_layer_weights(self):
		for index in range(len(self.layer)):
			self.layer[index].update_weights()

	def set_inputs(self, inputs):
		for index in range(len(self.layer)):
			self.layer[index].set_inputs(inputs[index])

