from sklearn.datasets import fetch_openml

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

boston = fetch_openml(name="boston", as_frame=True, version= 1)
X = boston.data
y = boston.target

model = LinearRegression()
model.fit(X, y)

print(model.coef_)


#test voi Rm(so phong)
X_rm = X[["RM"]].values
model.fit(X_rm,y)

w_0 = model.intercept_
w_1 = model.coef_[0]

x0 = np.linspace(X_rm.min(), X_rm.max(), 100)
y0 = w_0 + w_1 * x0

plt.plot(X_rm, y,'ro', alpha=0.3) #data
plt.plot(x0, y0, color='blue') #line

plt.xlabel("Average number of rooms (RM)")
plt.ylabel("House price")
plt.title("Boston Housing - Linear Regression (RM only)")

plt.show()