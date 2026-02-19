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


