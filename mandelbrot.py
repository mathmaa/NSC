"""
Mandelbrot set generator
Author : Mathias
Course : Numerical Scientific computing
"""
def mandelbrot_point(c, max_iter=100):
	z = 0
	for n in range(max_iter):
		z = z*z +c
		if abs(z) > 2:
			return n
	return max_iter



