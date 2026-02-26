"""
Mandelbrot set generator
Author : Mathias
Course : Numerical Scientific computing
"""
import numpy as np
import matplotlib.pyplot as plt
import time, statistics

def benchmark(func, *args, runs=3):
	times = []
	for _ in range(runs):
		start = time.perf_counter()
		result = func(*args)
		times.append(time.perf_counter() - start)
	median_time = statistics.median(times)
	print(f"Median: {median_time:.4f}s", f"(min={min(times):.4f}, max={max(times):.4f})")
	return median_time, result

def create_complex_grid(xmin=-2.0, xmax=1.0, 
						ymin=-1.5, ymax=1.5, 
						width=1024, height=1024):
	xvals=np.linspace(xmin, xmax, width)
	yvals=np.linspace(ymin, ymax, height)

	X, Y = np.meshgrid(xvals, yvals)

	C = X + 1j * Y
	return C

def compute_mandelbrot_set(C, max_iter=2000):
	Z = np.zeros_like(C)
	M = np.zeros(C.shape, dtype=int)
	for _ in range(max_iter):
		mask = np.abs(Z) <=2
		Z[mask] = Z[mask]**2 + C[mask]
		M[mask] += 1

	return M

xmin, xmax = -0.7487667139, -0.7487667078
ymin, ymax = 0.1236408449, 0.1236408510
width, height = 1024, 1024

C = create_complex_grid(xmin, xmax, ymin, ymax, width, height)

median_time, result = benchmark(compute_mandelbrot_set, C)

plt.imshow(result, extent=(-2, 1, -1.5, 1.5), cmap='hot')
plt.colorbar()
plt.title('Mandelbrot Set')
plt.savefig('mandelbrot.png')