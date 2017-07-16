import numpy as np
import csv
import numpy

class NN:

    def __init__(self, sl):

        #sl = number of units (not counting bias unit) in layer l
        self.sl = sl
        self.layers = len(sl)

        #Create weights
        self.weights = []
        for idx in range(1, self.layers):
            self.weights.append(numpy.matrix(numpy.random.rand(self.sl[idx-1]+1, self.sl[idx]))/5)

        self.cost = []

    def update(self, input):

        if input.shape[1] != self.sl[0]:
            raise ValueError, 'The first layer must have a node for every feature'

        self.z = []
        self.a = []

        #Input activations.  Expected inputs as numpy matrix (Examples x Featrues) 
        self.a.append(numpy.hstack((numpy.ones((input.shape[0], 1)), input)))#Set inputs ai + bias unit

        #Hidden activations
        for weight in self.weights: 
            self.z.append(self.a[-1]*weight)
            self.a.append(numpy.hstack((numpy.ones((self.z[-1].shape[0], 1)), 1/(1+numpy.exp(-self.z[-1]))))) #sigmoid

        #Output activation
        self.a[-1] = self.z[-1] #Not logistic regression thus no sigmoid function
        del self.z[-1]

    def backPropagate(self, targets, lamda):

        m = float(targets.shape[0]) #m is number of examples

        #Calculate cost
        Cost = -1/m*sum(numpy.power(self.a[-1] - targets, 2))
        for weight in self.weights:
            Cost = Cost + lamda/(2*m)*numpy.power(weight[1:, :], 2).sum()
        self.cost.append(abs(float(Cost)))

        #Calculate error for each layer
        delta = []
        delta.append(self.a[-1] - targets)
        for idx in range(1, self.layers-1): #No delta for the input layer because it is the input
            weight = self.weights[-idx][1:, :] #Ignore bias unit
            dsigmoid = numpy.multiply(self.a[-(idx+1)][:,1:], 1-self.a[-(idx+1)][:,1:]) #dsigmoid is a(l).*(1-a(l))
            delta.append(numpy.multiply(delta[-1]*weight.T, dsigmoid)) #Ignore Regularization

        Delta = []
        for idx in range(self.layers-1):
            Delta.append(self.a[idx].T*delta[-(idx+1)])

        self.weight_gradient = []
        for idx in range(len(Delta)):
            self.weight_gradient.append(numpy.nan_to_num(1/m*Delta[idx] + numpy.vstack((numpy.zeros((1, self.weights[idx].shape[1])), lamda/m*self.weights[idx][1:, :]))))

    def train(self, input, targets, alpha, lamda, iterations = 1000):

        #alpha: learning rate
        #lamda: regularization term

        for i in range(iterations):
            self.update(input)
            self.backPropagate(targets, lamda)
            self.weights = [self.weights[idx] - alpha*self.weight_gradient[idx] for idx in range(len(self.weights))]

    def autoparam(self, data, alpha = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3], lamda = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]):

        #data: numpy matrix with targets in last column
        #alpha: learning rate
        #lamda: regularization term

        #Create training, cross validation, and test sets
        while 1:
            try:
                numpy.seterr(invalid = 'raise')
                numpy.random.shuffle(data) #Shuffle data
                training_set = data[0:data.shape[0]/10*6, 0:-1]
                self.ntraining_set = (training_set-training_set.mean(axis=0))/training_set.std(axis=0)
                self.training_tgt = numpy.matrix(data[0:data.shape[0]/10*6, -1]).T

                cv_set = data[data.shape[0]/10*6:data.shape[0]/10*8, 0:-1]
                self.ncv_set = (cv_set-cv_set.mean(axis=0))/cv_set.std(axis=0)
                self.cv_tgt = numpy.matrix(data[data.shape[0]/10*6:data.shape[0]/10*8, -1]).T

                test_set = data[data.shape[0]/10*8:, 0:-1]
                self.ntest_set = (test_set-test_set.mean(axis=0))/test_set.std(axis=0)
                self.test_tgt = numpy.matrix(data[data.shape[0]/10*8:, -1]).T

                break

            except FloatingPointError:
                pass

        numpy.seterr(invalid = 'warn')
        cost = 999999
        for i in alpha:
            for j in lamda:
                self.__init__(self.sl)
                self.train(self.ntraining_set, self.training_tgt, i, j, 2000)
                current_cost = 1/float(cv_set.shape[0])*sum(numpy.square(self.predict(self.ncv_set) - self.cv_tgt)).tolist()[0][0]
                print current_cost
                if current_cost < cost:
                    cost = current_cost
                    self.learning_rate = i
                    self.regularization = j
        self.__init__(self.sl)

    def predict(self, input):

        self.update(input)
        return self.a[-1]


data = numpy.loadtxt(open('winequality-red.csv', 'rb'), delimiter = ',', skiprows = 1)#Load
numpy.random.shuffle(data)

features = data[:,0:10]
nfeatures = (features-features.mean(axis=0))/features.std(axis=0)
targets = numpy.matrix(data[:, 11]).T

n = NN([10, 30, 1])

n.train(nfeatures, targets, 0.07, 0.0, 100)

import matplotlib.pyplot
matplotlib.pyplot.subplot(221)
matplotlib.pyplot.plot(n.cost)
matplotlib.pyplot.title('Cost vs. Iteration')

matplotlib.pyplot.subplot(222)
matplotlib.pyplot.scatter(n.predict(nfeatures).tolist(), targets.tolist())
matplotlib.pyplot.plot(targets.tolist(), targets.tolist(), c = 'r')
matplotlib.pyplot.title('Data vs. Predicted')

matplotlib.pyplot.savefig('Report.png', format = 'png')
matplotlib.pyplot.close()
