import numpy
#import cupy as numpy
from time import perf_counter
from typing import List, Callable
import matplotlib.pyplot as plt
import json
from keras.datasets import mnist
from network import Network

from holo import prettyTime

(_Train_X, _Train_y), (_, _) = mnist.load_data()
# numpy shapes is like (lignes , colonnes)

PATH_NOISE = "./"


def timeit(func, *args):
    t = perf_counter()
    res = func(*args)
    return f"{perf_counter()-t} sec", res


def plot_list(lst: list, log=False):
    if log:
        plt.yscale('log')
    plt.plot(range(len(lst)), lst, label=nameof(lst=lst))
    plt.legend()
    plt.show()


def plot_list_multi(lsts, log=False):
    if log:
        plt.yscale('log')
    for i, lst in enumerate(lsts):
        plt.plot(range(len(lst)), lst, label=str(i))
    plt.legend()
    plt.show()


def plot_list_(log=False, **lsts):
    if log:
        plt.yscale('log')
    for name in lsts:
        plt.plot(range(len(lsts[name])), lsts[name], label=name)
    plt.legend()
    plt.show()


def smooth(lst, niv):
    box = numpy.ones(niv) / niv
    y_smooth = numpy.convolve(lst, box, mode='same')
    return y_smooth


def smooth_2(lst, niv):
    poly = numpy.poly1d(numpy.polyfit(range(len(lst)), lst, niv))
    print(poly)
    return poly(lst)


def nameof(**vars):
    return [x for x in vars]


def show(array):
    plt.imshow(array.reshape((28, 28)), cmap='gray', vmin=-1, vmax=1)


def save(network, path):
    with open(path + ".json", mode='w') as file:
        data = {}
        data["version"] = 2
        data["layering"] = network.layering
        data["w"] = [None] + [network.w[l].tolist() for l in range(1, network.L)]
        data["b"] = [None] + [network.b[l].tolist() for l in range(1, network.L)]
        json.dump(data, file)


def load_net(path):
    with open(path + ".json", mode='r') as file:
        data = json.load(file)
    assert data["version"] == 2
    return data["layering"], data["w"], data["b"]


def import_dataSet(path: str, size: int = -1) -> List[List[numpy.ndarray]]:
    dataset = []
    with open(path+".csv", mode='r', encoding='utf-8') as database:
        content = database.readlines()
        if size > -1:
            content = content[:size]
        for ligne in content:
            ligne = ligne.split(';')
            outputs = ligne[0]
            inputs = ligne[1][:-2]
            dataset.append([inputs, outputs])
    return rework_dataset(dataset)


def rework_dataset(dataset_row: List[List[str]]) -> List[List[numpy.ndarray]]:
    dataset = []
    for rowInputs, rowOutputs in dataset_row:
        inputs = numpy.array(rowInputs.split(
            ','), dtype=numpy.float64)  # .reshape(1,1024)
        inputs = inputs/255
        outputs = numpy.zeros((10), dtype=numpy.float64)
        outputs[int(rowOutputs)] = 1
        dataset.append([inputs, outputs])
    return dataset


def load_noises(path):
    with open(path+".json", mode='r') as file:
        data = json.load(file)
    assert data["version"] == 2
    prec = 10**(-data["prec"])
    return [numpy.array(data["noises"][i], dtype=numpy.float64)*prec for i in range(data["size"])]




def load_mnist(n=[60_000]):

    dataset = []
    for i in range(*n):
        outputs = numpy.zeros((10), dtype=numpy.float64)
        outputs[int(_Train_y[i])] = 1
        inputs = _Train_X[i].reshape(28*28)/255
        dataset.append([inputs, outputs])
    return dataset


def rConst(rate0, step): return rate0
def rLin(rate0, step, a=0.005):
    return rate0+a*step  # generalisation
def rPow_dec(rate0, step, a=0.20):
    return rate0/((step+1)**a)


def printSucessRate(network:Network):
    print(f"sucess rate:\n\ttrain: {network.evaluate_dataset(dataset_train):.4%}\n\ttest: {network.evaluate_dataset(dataset_test):.4%}")
    

def run(network:Network, rate0:float=4, steps:"int|tuple[int, int]"=75, rateMethode:"Callable[[float, int], float]"=rConst):
    printSucessRate(network)

    pack = numpy.arange(0, len(dataset_train), 1)
    packNoises = numpy.arange(0, len(noisePack), 1)
    t = perf_counter()
    network.cost_dataset(dataset_train)
    C = [network.get_linearCost()]
    R = [network.evaluate_dataset(dataset_test)]

    PROP = 50  # number of feed back on the curent ste of the net

    if isinstance(steps, tuple):
        start, stop = steps[0], steps[1]
        deltaSteps = stop-start
    else:
        deltaSteps = steps
        start = 0
        stop = steps
    
    print(rateMethode(rate0, stop))
    t0:float = perf_counter()
    step:"int|None" = None
    for step in range(start, stop):
        rate = rateMethode(rate0, step)
        if PACK:
            numpy.random.shuffle(pack)
            numpy.random.shuffle(packNoises)
            network.run_dataset_batch_noise(
                dataset_train, pack[:batchSize], packNoises, (1/1000 * NOISE_FORCE), rate, noisePack)
        else:
            network.run_dataset(dataset_train[:batchSize], rate)

        if (step-start) % (max(deltaSteps//PROP, 1)) == 0:
            C.append(network.get_linearCost())
            R.append(network.evaluate_dataset(dataset_test))
            print(f"step:{step-1:<4} learningRate:{rate:<10.4g} cost_1={C[-1]:<10.4g} correct test:{R[-1]:<10.2%}"
                  #f"(w,b) means: {network.get_w_abs()} "
                  f"{prettyTime((deltaSteps/(step+1-start)-1) * (perf_counter()-t))} remaining")
    
    print(prettyTime(perf_counter()-t0))

    network.cost_dataset(dataset_train)
    print(f"step:{step}:cost_1={network.get_linearCost():<22}")
    printSucessRate(network)
    return C, R

t = perf_counter()
try: 
    noisePack = load_noises(PATH_NOISE+"noise3")
    print(f"time to load the noise: {prettyTime(perf_counter()-t)}"); del t
except FileNotFoundError as err:
    print("failed to load cached noises, generating it on the fly")
    import noises_generator
    noisePack = noises_generator.create_noises((28, 28), 0.1, 10_000)
    print("created  0.1")
    print(f"time to generate the noise: {prettyTime(perf_counter()-t)}");   t = perf_counter()
    noises_generator.save_noises(PATH_NOISE + 'noise3.json', noisePack, prec=3)
    print(f"time to save the noise: {prettyTime(perf_counter()-t)}");   t = perf_counter()
    del t
print(f"size of noise pack: {len(noisePack)}")

batchSize = 128
dataset_train = load_mnist([len(noisePack)])
dataset_test = load_mnist([len(noisePack), len(noisePack)+2000])


pack = numpy.arange(0, len(dataset_train), 1)
PACK = True
# TODO faire une methode simplify qui entaine un nouveau reseau a reproduire des res aléatoires de celui a simplifier

NOISE_FORCE = 0.01
layer = [784, 64, 32, 32, 10]
#net1 = Network(*load_net(PATH_NET+'6')) # trained
net1 = Network(layer, wRate=0.5)
print(f"the network has {net1.get_num_w():_d} parameters")

l1, r1 = run(net1, rate0=0.5, steps=(0, 1000), rateMethode=rConst)

save(net1, path="./net1")

plot_list_(log=False, l1=l1)
plot_list_(log=False, r1=r1)

