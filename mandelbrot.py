"""
Mandelbrot set generator
Author : Mathias
Course : Numerical Scientific computing
"""
import numpy as np
from numba import njit

import time #, statistics

# def benchmark(func, *args, runs=3):
# 	times = []
# 	for _ in range(runs):
# 		start = time.perf_counter()
# 		result = func(*args)
# 		times.append(time.perf_counter() - start)
# 	median_time = statistics.median(times)
# 	print(f"Median: {median_time:.4f}s", f"(min={min(times):.4f}, max={max(times):.4f})")
# 	return median_time, result

def mandelbrot_naive(xmin=-2.0, xmax=1.0,
					 ymin=-1.5, ymax=1.5,
					 width=1024, height=1024,
					 max_iter=100):
	M = []
	for j in range(height):
		row = []
		y = ymin + (ymax - ymin) * j / (height - 1)
		for i in range(width):
			x = xmin + (xmax - xmin) * i / (width - 1)
			c = complex(x, y)
			z = 0j
			iteration = 0
			while abs(z) <= 2 and iteration < max_iter:
				z = z*z + c
				iteration += 1
			row.append(iteration)
		M.append(row)
	return M


def mandelbrot_numpy(xmin=-2.0, xmax=1.0,
					 ymin=-1.5, ymax=1.5,
					 width=1024, height=1024,
					 max_iter=100):
	xvals = np.linspace(xmin, xmax, width)
	yvals = np.linspace(ymin, ymax, height)
	X, Y = np.meshgrid(xvals, yvals)
	C = X + 1j * Y
	Z = np.zeros_like(C)
	M = np.zeros(C.shape, dtype=int)

	for _ in range(max_iter):
		mask = np.abs(Z) <= 2
		Z[mask] = Z[mask]**2 + C[mask]
		M[mask] += 1

	return M

@njit
def mandelbrot_point_numba(c, max_iter=100):
	z = 0j
	for n in range(max_iter):
		if z.real*z.real + z.imag*z.imag > 4.0:
			return n
		z = z*z + c
	return max_iter

def mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter=100):
	# outer loops still in Python
	x = np.linspace(xmin, xmax, width)
	y = np.linspace(ymin, ymax, height)
	result = np.zeros((height, width), dtype=np.int32)
	for i in range(height):
		for j in range (width):
			c = x[j] + 1j * y[i]
			result[i, j] = mandelbrot_point_numba(c, max_iter)
	return result

@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
	"""Fully JIT-compiled Mandelbrot --- structure identical to naive."""
	x = np.linspace(xmin, xmax, width)
	y = np.linspace(ymin, ymax, height)
	result = np.zeros((height, width), dtype=np.int32)
	for i in range(height):
		for j in range(width):
			c = x[j] + 1j * y[i]
			z = 0j # complex literal : type inference works !
			n = 0
			while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
				z = z * z + c
				n += 1
			result[i, j] = n
	return result

@njit
def mandelbrot_numba_typed(xmin, xmax, ymin, ymax, width, height, max_iter=100, dtype=np.float64):
	x = np.linspace(xmin, xmax, width).astype(dtype)
	y = np.linspace(ymin, ymax, height).astype(dtype)
	result = np.zeros((height, width), dtype=np.int32)
	for i in range(height):
		for j in range(width):
			c = x[j] + 1j * y[i]
			z = 0j # complex literal : type inference works !
			n = 0
			while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
				z = z * z + c
				n += 1
			result[i, j] = n
	return result

