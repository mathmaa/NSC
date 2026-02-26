"""
Mandelbrot set generator
Author : Mathias
Course : Numerical Scientific computing
"""
import numpy as np
from line_profiler import profile
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


def mandelbrot_numpy(xmin=-2.0, xmax=1.0, 
					ymin=-1.5, ymax=1.5, 
					width=1024, height=1024, 
					max_iter=100):
	
	xvals=np.linspace(xmin, xmax, width)
	yvals=np.linspace(ymin, ymax, height)
	X, Y = np.meshgrid(xvals, yvals)
	C = X + 1j * Y
	Z = np.zeros_like(C)
	M = np.zeros(C.shape, dtype=int)

	for _ in range(max_iter):
		mask = np.abs(Z) <=2
		Z[mask] = Z[mask]**2 + C[mask]
		M[mask] += 1

	return M