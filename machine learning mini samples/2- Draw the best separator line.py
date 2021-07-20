import numpy as np
import matplotlib.pyplot as plt

def sigmoid(score):
    return 1/(1+np.exp(-score))

def draw(x1, x2): 
    ln = plt.plot(x1, x2, '-')
    #plt.pause(0.001)   #این خط و خط پایین و جلو تب خورد و جلو رفتن درا برای دیدن مراخط خط کشی است
    #ln[0].remove()
    
# def calculate_error(lineParameters, points, y):
#     m = points.shape[0]
#     p = sigmoid(points * lineParameters)
#     cross_entropy = -(1/m) * (np.log(p).T * y + np.log(1-p).T * (1 - y))
#     return cross_entropy
    
def gradient_descent(line_parameters, points, y, alpha):
    m = points.shape[0]
    for i in range(20000):
        p = sigmoid(points* line_parameters)  #با توجه به فرمول
        gradient = (points.T * (p-y) * (alpha/m)) #(3,100) / (100,3) ترا نهاده
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b  = line_parameters.item(2)
        x1 = np.array([points[:,0].min(), points[:,0].max()])
        x2 = -b/w2+x1 * (-w1/w2)
    draw(x1,x2)
    #درا باید جلو بیاید
    #مقدار آلفا و اینرشن روی دقت تاثیر دارند

# np.random.seed(0)
n_pts = 10
bias = np.ones(n_pts)
a1= np.random.normal(10, 2, n_pts) 
a2 = np.random.normal(12, 2, n_pts)
b1 = np.random.normal(5, 2, n_pts)
b2 = np.random.normal(6, 2, n_pts)
top_region = np.array([a1, a2, bias]).T
bottom = np.array([b1, b2, bias]).T
all_point = np.vstack((top_region, bottom))
line_parameters = np.matrix([np.zeros(3)]).T
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)

_, ax= plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0], top_region[:, 1], color= 'r')
ax.scatter(bottom[:,0], bottom[:, 1], color= 'b')
gradient_descent(line_parameters, all_point, y, 0.006)
plt.show()
