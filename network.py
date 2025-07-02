import numpy
from timeit import timeit
print("py version")

def sigmoid(array):
    return 1 / (1+numpy.exp(-array))
def sigmoid_diff(array):
    exp_array = numpy.exp(-array)
    return exp_array/( (1+exp_array)**2)

def tanh(array):
    return numpy.tanh(array)
def tanh_diff(array):
    return 1 - numpy.tanh(array)**2

def relu(array):
    return numpy.maximum(0.0, array)
def relu_diff(array):
    return (array > 0) * 1.0

activFunc = tanh
activFunc_diff = tanh_diff

class Network():
    def __init__(self, layering:"list[int]", copy_w:"None|numpy.ndarray"=None,
            copy_b:"None|numpy.ndarray"=None, wRate:float=1.0)->None: # layering (nb neur on eache layer)
        self.layering = layering
        self.wRate = wRate
        self.L = len(self.layering)
        self.a = []
        self.b = [];  self.b_ = []
        self.z = [];  self.z_ = []
        self.w = [];  self.w_ = []
        self.activ = []
        self.C = -1.
        self.expectedOutput = numpy.zeros( (self.layering[-1]), dtype=numpy.float64 )
        for l, nbNeurones in enumerate(self.layering):
            self.a.append(numpy.zeros((nbNeurones), dtype=numpy.float64))
            if l!=0:
                self.z.append(numpy.zeros((nbNeurones), dtype=numpy.float64))
                self.z_.append(numpy.zeros((nbNeurones), dtype=numpy.float64))
                self.b_.append(numpy.zeros((nbNeurones), dtype=numpy.float64))
                self.w_.append(numpy.zeros((nbNeurones, self.layering[l-1]), dtype=numpy.float64))
                
                if copy_w is None:
                    self.w.append(numpy.random.uniform(-wRate, wRate, (nbNeurones, self.layering[l-1])))
                else:self.w.append(numpy.array(copy_w[l]))
                
                if copy_b is None:
                    self.b.append(numpy.random.uniform(-5*wRate, 5*wRate, (nbNeurones)))
                else:self.b.append(numpy.array(copy_b[l]))
            else:
                self.z.append(None);   self.w.append(None);   self.b.append(None)
                self.z_.append(None);  self.w_.append(None);  self.b_.append(None)

    def calc(self)->None:#step
        for l in range(1, self.L): 
            self.z[l] = self.w[l].dot(self.a[l-1])+self.b[l]
            if l == self.L-1:
                use_activ = sigmoid    
            else: use_activ = activFunc
            self.a[l] = use_activ(self.z[l])

    def cost(self, expectedOutput:numpy.ndarray)->None:
        assert expectedOutput.shape == self.expectedOutput.shape
        self.expectedOutput = expectedOutput
        self.C = numpy.sum( ( self.a[self.L-1]-self.expectedOutput )**2 )

    def set_inputs(self, inputs:numpy.ndarray)->None:
        assert inputs.shape == self.a[0].shape ,f"inputs shape{inputs.shape}  a[0] shape{self.a[0].shape}"
        self.a[0] = inputs

    def deepLearning_calc(self)->None:
        L = self.L-1
        factor:float = 1
        self.z_[L] = sigmoid_diff(self.z[L]) * 2 * (self.a[L]-self.expectedOutput)
        self.w_[L] += self.a[L-1]*self.z_[L].reshape(self.layering[L],1)
        self.b_[L] += self.z_[L]
        for l in range(L-1,0,-1):
            #factor *= 1.25 # manualy correct for vanishing gradient
            self.z_[l] = factor * activFunc_diff(self.z[l]) * numpy.sum(self.w[l+1] * self.z_[l+1].reshape(self.layering[l+1],1) ,axis=0)
            self.w_[l] += self.a[l-1]*self.z_[l].reshape(self.layering[l],1)
            self.b_[l] += self.z_[l]

    def apply_deeplearn(self, nbSamples:int, trainingRate:float)->None:
        for l in range(1,self.L):
            self.w[l] -= (self.w_[l] / nbSamples) * trainingRate # tester b learn en blockant w learn
            self.b[l] -= (self.b_[l] / nbSamples) * trainingRate

    def new_LearnStep(self)->None: # reset learning
        for l in range(1,self.L):
            self.w_[l].fill(0)
            self.b_[l].fill(0)
    
    def run_data_learn(self, inputs:numpy.ndarray,
                       expectedOutput:numpy.ndarray)->None:
        self.set_inputs(inputs)
        self.calc()
        self.cost(expectedOutput)
        self.deepLearning_calc()

    def run_data_learn_noise(self, inputs:numpy.ndarray, expectedOutput:numpy.ndarray, 
                             noise:numpy.ndarray, nF:float)->None:
        self.set_inputs(numpy.minimum( numpy.maximum( inputs+noise*nF ,0.) ,1.))
        self.calc()
        self.cost(expectedOutput)
        self.deepLearning_calc()

    def run_data(self, inputs:numpy.ndarray, expectedOutput:numpy.ndarray)->None:
        self.set_inputs(inputs)
        self.calc()
        self.cost(expectedOutput)

    def run_dataset(self, dataset:"list[list[numpy.ndarray]]", trainingRate:float)->None:
        N = len(dataset)
        avrgCost = 0
        self.new_LearnStep()#prep for new learn
        for inputs, expectedOutput in dataset:
            self.run_data_learn(inputs, expectedOutput)
            avrgCost += self.C
        self.apply_deeplearn(N, trainingRate)
        self.C = avrgCost / N

    def run_dataset_batch(self, dataset:"list[list[numpy.ndarray]]", 
                          pack:"list[int]", trainingRate:float)->None:
        N = len(pack)
        avrgCost = 0
        self.new_LearnStep()#prep for new learn
        for index in pack:
            self.run_data_learn(*dataset[int(index)])
            avrgCost += self.C
        self.apply_deeplearn(N, trainingRate)
        self.C = avrgCost / N

    def run_dataset_batch_noise(
            self, dataset:"list[list[numpy.ndarray]]",
            pack:"list[int]", packNoies:"list[int]", 
            nF:float, trainingRate:float, noisePack)->None:
        N = len(pack)
        avrgCost = 0
        self.new_LearnStep()#prep for new learn
        for index in pack:
            self.run_data_learn_noise(*dataset[int(index)], packNoies[int(index)], nF)
            avrgCost += self.C
        self.apply_deeplearn(N, trainingRate)
        self.C = avrgCost / N

    def cost_dataset(self, dataset:"list[list[numpy.ndarray]]")->None:
        N = len(dataset)
        avrgCost = 0
        for inputs, expectedOutput in dataset:
            self.run_data(inputs, expectedOutput)
            avrgCost += self.C
        self.C = avrgCost / N

    def test_on_data(self, inputs:numpy.ndarray, outputs:numpy.ndarray)->None:
        timeit(self.run_data, inputs, outputs)
        print(f"lin_Cost={self.get_linearCost():<22}, Cost={self.C:<22}")
        print(f"predict={self.a[-1].argmax():<6}, res={outputs.argmax():<6}")

    def evaluate_dataset(self, dataset:"list[list[numpy.ndarray]]")->float:
        N = len(dataset)
        avrgSucess = 0
        avrgCost = 0
        for inputs, expectedOutput in dataset:
            self.run_data(inputs, expectedOutput)
            if self.a[-1].argmax() == expectedOutput.argmax():
                avrgSucess += 1
            avrgCost += self.C
        self.C = avrgCost / N
        return avrgSucess / N


    def export(self)->"tuple[list[int], list[None|numpy.ndarray]]":
        return (self.layering.copy(), [None]+[layer.copy() for layer in self.w[1:]])

    def get_linearCost(self)->float:#0<=x<=1
        return (self.C / 10) ** 0.5

    def get_result(self)->int:
        return self.a[-1]

    def get_w_abs(self)->"list[str]":
        return [f"({numpy.abs(w).mean():.4g}, {numpy.abs(b).mean():.4g})" for w, b in zip(self.w[1:], self.b[1:])]

    def get_num_w(self)->int:
        return sum([self.layering[i] * self.layering[i-1] for i in range(1,self.L)])
