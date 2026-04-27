import numpy as np
import time
import statistics
from numba import njit
from multiprocessing.pool import Pool
from typing import Dict, Optional, List, Tuple
from dask import delayed

def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
	"""
	Compute the escape iteration count for a single point in the Mandelbrot set.

	Parameters:
		c_real, c_imag : float
			Real and imaginary parts of the complex number c.
		max_iter : int
			Maximum number of iterations.

	Returns:
		int:
			Number of iterations before divergence, or max_iter if the
			point does not escape.
	"""
	z_real = z_imag = 0.0
	for i in range(max_iter):
		zr2 = z_real * z_real
		zi2 = z_imag * z_imag
		if zr2 + zi2 > 4.0: 
			return i
		z_imag = 2.0 * z_real * z_imag + c_imag
		z_real = zr2 - zi2 + c_real
	return max_iter

@njit
def mandelbrot_pixel_numba(c_real: float, c_imag: float, max_iter: int) -> int:
	"""
	Numba-accelerated version of mandelbrot_pixel.

	Parameters:
		c_real, c_imag : float
		max_iter : int

	Returns:
		int:
			Escape iteration count.
	"""
	z_real = z_imag = 0.0
	for i in range(max_iter):
		zr2 = z_real * z_real
		zi2 = z_imag * z_imag
		if zr2 + zi2 > 4.0: 
			return i
		z_imag = 2.0 * z_real * z_imag + c_imag
		z_real = zr2 - zi2 + c_real
	return max_iter

@njit
def mandelbrot_chunk(
	row_start: int, row_end: int,
	N: int,
	x_min: float, x_max: float,
	y_min: float, y_max: float,
	max_iter: int
) -> np.ndarray:
	"""
	Compute a contiguous chunk of rows of the Mandelbrot set.

	Returns:
		np.ndarray: Subgrid of shape (row_end - row_start, N).
	"""
	out = np.empty((row_end - row_start, N), dtype=np.int32)
	dx = (x_max - x_min) / (N - 1)
	dy = (y_max - y_min) / (N - 1)

	for r in range(row_end - row_start):
		c_imag = y_min + (r + row_start) * dy
		for col in range(N):
			out[r, col] = mandelbrot_pixel_numba(
				x_min + col * dx, c_imag, max_iter
			)
	return out

def mandelbrot_serial(
	N: int,
	x_min: float, x_max: float,
	y_min: float, y_max: float,
	max_iter: int = 100
) -> np.ndarray:
	"""
	Compute the full Mandelbrot grid serially using a single chunk.
	"""
	return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def _worker(args: Tuple) -> np.ndarray:
	"""
	Worker wrapper for multiprocessing; unpacks arguments for chunk computation.
	"""
	return mandelbrot_chunk(*args)

def mandelbrot_parallel(
	N: int,
	x_min: float, x_max: float,
	y_min: float, y_max: float,
	max_iter: int = 100,
	n_workers: int = 1,
	n_chunks: Optional[int] = None,
	pool: Optional[Pool] = None
) -> np.ndarray:
	"""
	Compute the Mandelbrot set in parallel using multiprocessing.

	Splits the grid into row chunks and distributes them across workers.

	Returns:
		np.ndarray: Full grid assembled from computed chunks.
	"""
	if n_chunks is None:
		n_chunks = n_workers

	chunk_size = max(1, N // n_chunks)
	chunks: List[Tuple] = []
	row = 0

	while row < N:
		row_end = min(row + chunk_size, N)
		chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
		row = row_end

	if pool is not None:
		return np.vstack(pool.map(_worker, chunks))

	# Warm-up (optional, avoids startup overhead in timing contexts)
	tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]

	with Pool(processes=n_workers) as p:
		p.map(_worker, tiny)
		parts = p.map(_worker, chunks)

	return np.vstack(parts)

def mandelbrot_dask(
	N: int,
	x_min: float, x_max: float,
	y_min: float, y_max: float,
	max_iter: int = 100,
	n_chunks: int = 32
) -> np.ndarray:
	"""
	Compute the Mandelbrot set using Dask delayed tasks.

	Splits the grid into row chunks and evaluates them lazily,
	then combines results with np.vstack.
	"""
	chunk_size = max(1, N // n_chunks)
	tasks = []

	for row_start in range(0, N, chunk_size):
		row_end = min(row_start + chunk_size, N)

		task = delayed(mandelbrot_chunk)(
			row_start, row_end,
			N,
			x_min, x_max,
			y_min, y_max,
			max_iter
		)
		tasks.append(task)

	results = delayed(np.vstack)(tasks)
	return results.compute()

def benchmark(
	N: int = 1024,
	x_min: float = -2, x_max: float = 1,
	y_min: float = -1.5, y_max: float = 1.5,
	max_iter: int = 100,
	worker_counts: Optional[List[int]] = None,
	runs: int = 3
) -> Dict[str, object]:
	"""
	Benchmark serial and multiprocessing Mandelbrot implementations.

	Returns:
		dict:
			{
				"serial": float,
				"parallel": {workers: float, ...}
			}
	"""
	if worker_counts is None:
		worker_counts = [1, 2, 4, 8]

	print("Warming up numba...")
	mandelbrot_serial(64, x_min, x_max, y_min, y_max, max_iter)
	print("Warmed up\n")

	results: Dict[str, object] = {}

	# Serial
	print("Benchmarking serial")
	times: List[float] = []

	for _ in range(runs):
		t0 = time.perf_counter()
		mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
		times.append(time.perf_counter() - t0)

	median_time = statistics.median(times)
	results["serial"] = median_time
	print(f"Median serial: {median_time:.4f}s (min={min(times):.4f}, max={max(times):.4f})")

	# Parallel
	times_parallel: Dict[int, float] = {}

	for n in worker_counts:
		print(f"Benchmarking parallel with {n} workers\n")
		times = []

		for _ in range(runs):
			t0 = time.perf_counter()
			mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers=n)
			times.append(time.perf_counter() - t0)

		median_time = statistics.median(times)
		times_parallel[n] = median_time

		print(f"Median parallel_{n}: {median_time:.4f}s (min={min(times):.4f}, max={max(times):.4f})")

	results["parallel"] = times_parallel

	return results