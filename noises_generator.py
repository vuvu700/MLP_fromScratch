import json
import numpy 
from opensimplex import OpenSimplex
import matplotlib.pyplot as plt
import time

PATH_NOISE=r"./"

def import_dataSet(path:str, size:int=-1)->"list[list[str]]":
    dataset = []
    with open(path, mode='r', encoding='utf-8') as database:
        content = database.readlines()
        #print(len(content))
        if size > -1:
            content = content[: size]
        for ligne in content:
            ligne = ligne.split(';')
            outputs = ligne[0]
            inputs = ligne[1][: -2]
            dataset.append([inputs, outputs])
    return dataset

def rework_dataset(dataset_row:"list[list[str]]")->"list[list[numpy.ndarray]]":
    dataset = []
    for rowInputs, rowOutputs in dataset_row:
        inputs = numpy.array(rowInputs.split(','),dtype=numpy.float64).reshape(1,1024)
        inputs = inputs / 255
        outputs = numpy.zeros((1, 10), dtype=numpy.float64)
        outputs[0, int(rowOutputs)] = 1
        dataset.append([inputs, outputs])
    return dataset

def create_noises(shape, step, number):
    noisePack = []
    noise = OpenSimplex()
    (sx, sy) = shape
    for _ in range(number):
        (deltax, deltay) = numpy.random.randint(-1000, 1000, size=2)
        lst = numpy.array([noise.noise2(deltax + (i//sx)*step, deltay + (i%sy)*step) for i in range(sx*sy)], dtype=numpy.float64)
        noisePack.append(lst)
    return noisePack

def save_noises(path, noisePack, prec=3):
    data = {}
    data["version"] = 2
    data["prec"] = prec
    data['size'] = len(noisePack)
    data['noises'] = [numpy.int64(noisePack[i] * 10**prec).tolist() for i in range(data['size'])]
    with open(path, mode='w') as file:
        json.dump(data, file, separators=(',', ':'))

def load_noises(path):
    with open(path, mode='r') as file:
        data = json.load(file)
    assert data["version"] == 2
    prec = 10**(-data["prec"])
    return [numpy.array(data["noises"][i], dtype=numpy.float64) * prec for i in range(data["size"])]


def show(array):
    #plt.pcolor(X, Y, v, cmap=cm)
    #plt.clim(-1,1)
    #plt.show()
    plt.imshow(array.reshape((28,28)),cmap='gray',vmin=-1,vmax=1)

def fusion(iNoise,iDataset,coef):
    return numpy.minimum(
        numpy.maximum(
            dataset2[iDataset][0] + noises[iNoise] * coef,
            0.), 
        1.)


if __name__ == "__main__":
    t = time.perf_counter()

    #faire en sorte de pouvoir append les fichiers en import les f puis fusion puis save 

    #noises = create_noises((28, 28), 2, 1024)
    #print("created  2")
    #print(time.perf_counter()-t);   t = time.perf_counter()
    #save_noises(PATH_NOISE + '1.json', noises, prec=3)
    #print(time.perf_counter()-t);   t = time.perf_counter()
    #print("saved 2")
    #
    #noises = create_noises((28,28),0.5,1024)
    #print("created  0.5")
    #print(time.perf_counter()-t);   t = time.perf_counter()
    #save_noises(PATH_NOISE + '2.json', noises, prec=3)
    #print(time.perf_counter()-t);   t = time.perf_counter()
    #print("saved 0.5")

    noises = create_noises((28, 28), 0.1, 10_000)
    print("created  0.1")
    print(time.perf_counter()-t);   t = time.perf_counter()
    save_noises(PATH_NOISE + 'noise3.json', noises, prec=3)
    print(time.perf_counter()-t);   t = time.perf_counter()
    print("saved 0.1")

