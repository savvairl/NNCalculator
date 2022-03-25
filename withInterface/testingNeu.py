from pybrain3.tools.xml import networkreader
net = networkreader.NetworkReader.readFrom('network.xml')

print('loaded:', net.activate([9, 9]))