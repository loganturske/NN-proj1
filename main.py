from Layer import Layer
from Perceptron import Perceptron
from Network import Network

inputs = [[1,2],[1,2]]
weights = [[.3,.3],[.3,.3]]
weights2 = [[.8,.8]]

desired_output = .7
bias = [0, 0]
eta = 1

first_io = [[1,1],[1,1]]
first_desired_output = .9
second_io = [[-1,-1],[-1,-1]]
second_desired_output = .05

l1 = Layer(False, 2, 2, weights, bias, first_desired_output, eta) 


l2 = Layer(True, 2, 1, weights2, bias, first_desired_output, eta)

net = Network(l1,l2)

# net.feedforward(inputs, .7)
# net.backpropogate()
# net.feedforward(inputs, .7)
# net.backpropogate()

## Method 1 ##
# for index in range(15):
# 	net.feedforward(first_io, first_desired_output)
# 	net.backpropogate()
# 	net.feedforward(second_io, second_desired_output)
# 	net.backpropogate()

# Method 2 ##
for index in range(15):
	net.feedforward(first_io, first_desired_output)
	net.backpropogate()

for index in range(15):
	net.feedforward(second_io, second_desired_output)
	net.backpropogate()
	
# net.feedforward(first_io, first_desired_output)
net.feedforward(second_io, second_desired_output)

net.print()