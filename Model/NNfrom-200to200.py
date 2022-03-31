import numpy as np

from pybrain3.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain3.datasets import SupervisedDataSet
train_data = SupervisedDataSet(2, 1)

# Подготовка исходных данных
x1 = np.array([])
x2 = np.array([])
# 0 to 200
for i in range(1000):
    for j in range(1000):
        x1 = np.append(x1, i/10)
        x2 = np.append(x2, j/10)
# 200 to 0
for i in range(1000):
    for j in range(1000):
        x1 = np.append(x1, i/10)
        x2 = np.append(x2, -j/10)
# 0 to -200
for i in range(1000):
    for j in range(1000):
        x1 = np.append(x1, -i/10)
        x2 = np.append(x2, -j/10)

x = np.append(x1, x2)
x = np.reshape(x, [len(x1), 2])
y = np.transpose(sum(np.transpose(x)))
for i in range(len(x1)):
    print(x[i,0],'+',x[i,1],'=',y[i])

for i in range(len(x1)):
    train_data.addSample(x[i, :], y[i])



print(train_data)
import math
# Создание и Обучение нейронной сети
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.supervised.trainers import BackpropTrainer

# Создание нейронной сети с двумя входами, двумя скрытыми слоями по 3 нейрона и одним выходом:
nn = buildNetwork(2, 3,3, 1, bias=True, hiddenclass=SigmoidLayer)
# Количество элементов передаваемых в сеть должно быть равно количеству входов.
# Метод возвращает ответ в виде единственного числа, если текущая цепь имеет один выход


trainer = BackpropTrainer(nn, train_data)

trainer.trainEpochs(300)

print('training')
print(trainer.testOnData(dataset=train_data))

print(nn.activate([2,3]))
print(nn.activate([0.5,2.5]))
for i in range(len(x1)):
    print(x1[i], '+', x2[i], '=', nn.activate([x1[i],x2[i]]))

from pybrain3.tools.xml import networkreader
from pybrain3.tools.xml import networkwriter

networkwriter.NetworkWriter.writeToFile(nn, 'network.xml')
net = networkreader.NetworkReader.readFrom('network.xml')


print('loaded:', net.activate([8, 3]))    # значения