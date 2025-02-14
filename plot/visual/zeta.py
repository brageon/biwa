'''
Real part is 1/2 just like logical contradictions for example Liar paradox. Compare this with gravity and coloumb distance formulas.
'''
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
def riemann_zeta(s, meshgrid_points=1000):
  x = np.linspace(-10, 10, meshgrid_points)
  y = np.linspace(-10, 10, meshgrid_points)
  X, Y = np.meshgrid(x, y)
  Z = X + 1j * Y
  return Z, np.log(Z**s)
s = complex(0.1, 2.0) 
Z, result = riemann_zeta(s) 
print(f"The value of Zeta({s}) is: {result}")
plt.contourf(Z.real, Z.imag, result.imag, levels=200, cmap='hot')
plt.colorbar()
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title('Complex logarithm of Riemann Zeta function')
plt.show()
