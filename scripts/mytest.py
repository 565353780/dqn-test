import matplotlib.pyplot as plt
import numpy as np

skip_num = 1000

with open('data.txt', 'r') as f:
    lines = f.readlines()

    line_size = int(len(lines) / skip_num)

    data = np.zeros((line_size, 2))

    for i in range(line_size):
        data[i][0] = float(lines[skip_num * i].split(',')[0].split('{')[1])
        data[i][1] = float(lines[skip_num * i].split(',')[1].split('}')[0])

    plt.plot(data[:, 0], data[:, 1])
    plt.show()
