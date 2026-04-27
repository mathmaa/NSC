"""
Mandelbrot set generator
Author : Mathias
Course : Numerical Scientific computing
"""
import numpy as np
from numba import njit

def mandelbrot_naive(
	xmin: float=-2.0, xmax: float=1.0,
	ymin: float=-1.5, ymax: float=1.5,
	width: int=1024, height: int=1024,
	max_iter: int=100
) -> list[list[int]]:
	"""
	Compute a Mandelbrot set grid using a naive nested-loop approach.

	Parameters:
		xmin, xmax : float
			Range of the real axis.
		ymin, ymax : float
			Range of the imaginary axis.
		width, height : int
			Resolution of the output grid.
		max_iter : int
			Maximum number of iterations.

	Returns:
		list[list[int]]: 2D grid of iteration counts.
	"""
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

def mandelbrot_numpy(
	xmin: float=-2.0, xmax: float=1.0,
	ymin: float=-1.5, ymax: float=1.5,
	width: int=1024, height: int=1024,
	max_iter: int=100
) -> np.ndarray:
	"""
	Compute the Mandelbrot set using a vectorized NumPy implementation.

	This function evaluates the escape-time algorithm over a grid in the
	complex plane using array operations instead of explicit Python loops.

	Parameters:
		xmin, xmax : float
			Range of the real axis.
		ymin, ymax : float
			Range of the imaginary axis.
		width, height : int
			Resolution of the output grid.
		max_iter : int
			Maximum number of iterations.

	Returns:
		np.ndarray:
			2D array (height × width) of iteration counts.
	"""
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
def mandelbrot_point_numba(
	c: complex, max_iter: int=100
) -> int:
	"""
	Compute the escape iteration count for a single complex point
	using a Numba-accelerated implementation.

	Parameters:
		c : complex
			Complex number representing a point in the plane.
		max_iter : int
			Maximum number of iterations.

	Returns:
		int:
			Number of iterations before divergence, or max_iter if the
			point does not escape.
	"""
	z = 0j
	for n in range(max_iter):
		if z.real*z.real + z.imag*z.imag > 4.0:
			return n
		z = z*z + c
	return max_iter

def mandelbrot_hybrid(
	xmin: float, xmax: float,
	ymin: float, ymax: float,
	width: int, height: int,
	max_iter: int = 100
) -> np.ndarray:
	"""
	Compute the Mandelbrot set using a hybrid approach:
	Python loops with a Numba-accelerated point function.

	The grid is generated using NumPy, while each point is evaluated
	via `mandelbrot_point_numba`, which is JIT-compiled.

	Parameters:
		xmin, xmax : float
			Range of the real axis.
		ymin, ymax : float
			Range of the imaginary axis.
		width, height : int
			Resolution of the output grid.
		max_iter : int
			Maximum number of iterations.

	Returns:
		np.ndarray:
			2D array (height × width) of iteration counts.
	"""
	x = np.linspace(xmin, xmax, width)
	y = np.linspace(ymin, ymax, height)
	result = np.zeros((height, width), dtype=np.int32)
	for i in range(height):
		for j in range (width):
			c = x[j] + 1j * y[i]
			result[i, j] = mandelbrot_point_numba(c, max_iter)
	return result

@njit
def mandelbrot_naive_numba(
	xmin: float, xmax: float,
	ymin: float, ymax: float,
	width: int = 1024, height: int = 1024,
	max_iter: int = 100
) -> np.ndarray:
	"""
	Compute the Mandelbrot set using a fully Numba JIT-compiled implementation.

	This version mirrors the naive nested-loop algorithm but is entirely
	compiled with Numba, including grid generation and iteration logic.

	Parameters:
		xmin, xmax : float
			Range of the real axis.
		ymin, ymax : float
			Range of the imaginary axis.
		width, height : int
			Resolution of the output grid.
		max_iter : int
			Maximum number of iterations.

	Returns:
		np.ndarray:
			2D array (height × width) of iteration counts.
	"""
	
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
def mandelbrot_numba_typed(
	xmin: float, xmax: float,
	ymin: float, ymax: float,
	width: int, height: int,
	max_iter: int = 100,
	dtype: np.dtype = np.float64
) -> np.ndarray:
	"""
	Compute the Mandelbrot set using a fully JIT-compiled Numba implementation
	with configurable floating-point precision.

	Parameters:
		xmin, xmax : float
		ymin, ymax : float
			Bounds of the complex plane.
		width, height : int
			Grid resolution.
		max_iter : int
			Maximum number of iterations.
		dtype : numpy dtype
			Floating-point precision for computations (e.g., np.float64).

	Returns:
		np.ndarray:
			2D array of iteration counts (height × width).
	"""
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