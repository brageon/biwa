'''
Root mean square measure the dispersion or variability of a dataset. A set of exam scores can be used to compare the overall difficulty of different exams.  (2860, 2872) (3754, 3766) (2974, 2986), (3117, 3129). Integral here is used to define trapezoid functions from index positions by gaps with topology. 
'''
import numpy as np
def find_integral(data):
    integral = 0
    for i in range(len(data) - 1):
        current_value = data[i][1]
        next_value = data[i + 1][1]
        width = next_value - current_value
        midpoint = (current_value + next_value) / 2
        integral += midpoint * width
    return integral
data = [(511, 523), (1117, 1129), (1771, 1783),]
integral = find_integral(data)
rms = np.sqrt(integral)
print(rms)                                                                             
