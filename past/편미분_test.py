import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-3, 3, 0.3)
y = np.arange(-3, 3, 0.3)
#z = (x ** 2) + (y ** 2)
z = np.array([])

for i in range(0, len(x)) :
    z = np.append(z, (x[i] ** 2) * (y ** 2), axis = 0)

print(z)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z)
plt.show()
