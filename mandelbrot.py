"""
Mandelbrot set generator
Author : Mathias
Course : Numerical Scientific computing
"""
import numpy as np

def mandelbrot_point(c, max_iter=100):
	z = 0
	for n in range(max_iter):
		z = z*z +c
		if abs(z) > 2:
			return n
	return max_iter

print(mandelbrot_point(complex(0, 0)))
print(mandelbrot_point(complex(1, 1)))
print(mandelbrot_point(complex(-2, 3)))  

def compute_mandelbrot_set(xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5, width=100, height=100, max_iter=100):
	xvalues = np.linspace(xmin, xmax, width)
	yvalues = np.linspace(ymin, ymax, height)

	result = np.zeros((height, width), dtype=int)
	for i in range(height):
		for j in range(width):
			c = complex(xvalues[j], yvalues[i])
			result[i, j] = mandelbrot_point(c, max_iter)

	return result

grid = compute_mandelbrot_set(100, 100, max_iter=100)
print(grid.shape)  # Should be (100, 100)
