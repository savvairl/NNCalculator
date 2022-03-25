import numpy as np

# Подготовка исходных данных

x1 = np.array([])
x2 = np.array([])
for i in range(100):
    for j in range(100):
        x1 = np.append(x1, i/10)
        x2 = np.append(x2, j/10)

x = np.append(x1, x2)
x = np.reshape(x, [len(x1), 2])
y = np.transpose(sum(np.transpose(x)))
for i in range(len(x1)):
    print(x[i,0],'+',x[i,1],'=',y[i])

from pybrain3.datasets import SupervisedDataSet
train_data = SupervisedDataSet(2, 1)

for i in range(len(x1)):
    train_data.addSample(x[i, :], y[i])

print(train_data)

# Создание и Обучение нейронной сети
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.structure import TanhLayer
from pybrain3.supervised.trainers import BackpropTrainer

# Создание нейронной сети с двумя входами, двумя скрытыми слоями и одним выходом:
nn = buildNetwork(2, 15, 1, bias=True, hiddenclass=TanhLayer)
# Количество элементов передаваемых в сеть должно быть равно количеству входов.
# Метод возвращает ответ в виде единственного числа, если текущая цепь имеет один выход


trainer = BackpropTrainer(nn, train_data)

for epoch in range(100):
   trainer.train()

print('training')
print(trainer.testOnData(dataset=train_data))


for i in range(len(x1)):
    print(x1[i], '+', x2[i], '=', nn.activate(x[i,:]))

from pybrain3.tools.xml import networkreader
from pybrain3.tools.xml import networkwriter

networkwriter.NetworkWriter.writeToFile(nn, 'network.xml')
net = networkreader.NetworkReader.readFrom('network.xml')


print('loaded:', net.activate([8, 3]))    # значения