import activation

from parameter import params

class Layer:
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2
    TEACHING = 3
    BIAS = 4
    
    
class Neuron:
    def __init__(self, id=0, layer=None, act_fn=activation.sigmoid):
        """
        Create a simple base neuron i.e. a node
        """
        self.id = id
        self.layer = layer
        
        self.activation = act_fn
        if self.layer == 0:
            self.activation = params['input_act_fn']
        elif self.layer == 1:
            self.activation = params['hidden_act_fn']
        elif self.layer == 2:
            self.activation = params['output_act_fn']
        elif self.layer == 3:
            self.activation = params['output_act_fn']
        elif self.layer == 4:
            self.activation = params['input_act_fn']
            
        if self.activation == activation.sigmoid:
            self.derivative = activation.sigmoid_prime
        elif self.activation == activation.tanh:
            self.derivative = activation.tanh_prime
        elif self.activation == activation.relu:
            self.derivative = activation.relu_prime
        elif self.activation == activation.linear:
            self.derivative = activation.linear_prime
        elif self.activation == activation.softmax:
            self.derivative = activation.softmax_prime
        
        #save all outcoming Connections
        self.outConnections = []

        #calculation Values
        self.inputActivation = None
        self.outputActivation = None
        
        # pre- and post- synaptic weights
    #----------------------------------------------------------------------------------

    def clear(self):

        self.inputActivation = 0
        self.outputActivation = 0

    #----------------------------------------------------------------------------------

    def addActivation(self, x):
        
        self.inputActivation += x
        
    #----------------------------------------------------------------------------------

    def calculate(self):

        #apply Sigmoid activation to all Layers except the input Layer
        self.outputActivation = self.activation(self.inputActivation) if self.layer > 0 else self.inputActivation
        
    #----------------------------------------------------------------------------------

    def passToNextLayer(self):

        for connection in self.outConnections:
            if connection.expressed:
                connection.outNode.addActivation(self.outputActivation * connection.weight)
                
    #----------------------------------------------------------------------------------
    
    def clone(self):

        return Neuron(self.id, self.layer)
    
    def __repr__(self) -> str:
        STRING = "{}: {} (Layer {})"
        return STRING.format(self.name(), self.id, self.layer)
    
    def name(self) -> str:
        """
        Returns the name of the class.
        :return: Class name
        """
        return self.__class__.__name__
        
    