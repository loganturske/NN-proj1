class Network:

	def __init__(self, hidden_layer, output_layer):
		self.hidden_layer = hidden_layer
		self.output_layer = output_layer

	def feedforward(self, inputs, desired_output):
		self.hidden_layer.set_inputs(inputs)
		self.hidden_layer.set_desired_output(desired_output)

		hidden_output = []
		hidden_output.append(self.hidden_layer.get_layer_output_vector())

		self.output_layer.set_desired_output(desired_output)
		self.output_layer.set_inputs(hidden_output)
		self.output_layer.get_layer_output_vector()
		self.output_layer.get_error_vector()

	def backpropogate(self):
		self.output_layer.calc_layer_delta_weights()
		self.output_layer.set_hidden_layer_delta_values(self.hidden_layer)
		self.hidden_layer.udpate_layer_weights()

	def print(self):
		print("### HIDDEN LAYER ###")
		self.hidden_layer.print()
		print("### OUTPUT LAYER ###")
		self.output_layer.print()
		print("Big E: " + str(self.output_layer.get_big_E()))

	def print_big_E(self):
		print("Big E: " + str(self.output_layer.get_big_E()))
