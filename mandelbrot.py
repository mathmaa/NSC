"""
Mandelbrot set generator
Author : Mathias
Course : Numerical Scientific computing
"""
import numpy as np
import matplotlib.pyplot as plt
import time

def mandelbrot_point(c, max_iter=100):
	z = 0
	for n in range(max_iter):
		z = z*z +c
		if abs(z) > 2:
			return n
	return max_iter

def compute_mandelbrot_set(xmin=-2.0, xmax=1.0, 
						   ymin=-1.5, ymax=1.5, 
						   width=100, height=100, 
						   max_iter=100):
	xvalues = np.linspace(xmin, xmax, width)
	yvalues = np.linspace(ymin, ymax, height)

	result = np.zeros((height, width), dtype=int)
	for i in range(height):
		for j in range(width):
			c = complex(xvalues[j], yvalues[i])
			result[i, j] = mandelbrot_point(c, max_iter)

	return result

start = time.time() 
result = compute_mandelbrot_set(xmin=-2.0, xmax=1.0, 
								ymin=-1.5, ymax=1.5, 
								width=1024, height=1024)
time_taken = time.time() - start
print(f"Time taken to compute Mandelbrot set: {time_taken:.2f} seconds")

plt.imshow(result, extent=(-2, 1, -1.5, 1.5), cmap='hot')
plt.colorbar()
plt.title('Mandelbrot Set')
plt.savefig('mandelbrot.png')