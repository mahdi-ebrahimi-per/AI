import matplotlib.pyplot as plt
import numpy as np

def draw(x1, x2):
    ln = plt.plot(x1, x2)

def sigmoid(score):
    return 1/(1+np.exp(-score))
np.random.seed(1)
n_pts = 10
bias = np.ones(n_pts)
# print(bias)
a1= np.random.normal(10, 2, n_pts)
a2 = np.random.normal(12, 2, n_pts)
b1 = np.random.normal(5, 2, n_pts)
b2 = np.random.normal(6, 2, n_pts)
top_region = np.array([a1, a2, bias]).T
bottom = np.array([b1, b2, bias]).T

all_point = np.vstack((top_region, bottom))

w1 = -0.2
w2 = -0.35
b = 3.5  #bias

line_parameters = np.matrix([w1,w2,b]).T

print(all_point.shape)
print(line_parameters.shape)

linear_combination = all_point * line_parameters
probabilities = sigmoid(linear_combination)
# print(probabilities)
print(bottom[:,0].min())
print(top_region[:,0].max())
x1 = np.array([bottom[:,0].min(), top_region[:,0].max()])
x2 = -b/w2 + x1 * (-w1/w2)
_, ax= plt.subplots(figsize=(5,5))
ax.scatter(top_region[:,0], top_region[:, 1], color= 'r')
ax.scatter(bottom[:,0], bottom[:, 1], color= 'b')
draw(x1,x2)
plt.show()
