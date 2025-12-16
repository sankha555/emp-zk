import numpy as np

# constants
DIV1 = 255.0
SUB  = 0.1307000070810318
DIV2 = 0.30810001492500305

# read CSV (one row per line, comma-separated)
data = np.loadtxt("test/ai/data/inputs/mnist_test.csv", delimiter=",")
print(data.shape)

# transform
new_data = []
for i, x in enumerate(data):
    
    dp = [x[0]]
    for j in x[1:]:
        dp.append((j/DIV1 - SUB)/DIV2)
        
    new_data.append(dp)

# write to text file (space-separated, one row per line)
np.savetxt("test/ai/data/inputs/mnist_conv.txt", new_data, fmt="%.6f")