import numpy as np
from numba import njit
from multiprocessing import Pool
import time
import statistics

from dask import delayed


@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
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
def mandelbrot_chunk(row_start, row_end, 
                     N, 
                     x_min, x_max, 
                     y_min, y_max, 
                     max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1) 
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r,col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)
            
    return out

def mandelbrot_serial(N, 
                      x_min, x_max, 
                      y_min, y_max, 
                      max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, 
                        x_min, x_max, 
                        y_min, y_max, 
                        max_iter=100,
                        n_workers=1,
                        n_chunks=None,
                        pool=None):
    if n_chunks is None:
        n_chunks = n_workers

    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0

    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
        
    if pool is not None:
        return np.vstack(pool.map(_worker, chunks))
    tiny = [(0,8,8,x_min,x_max,y_min,y_max,max_iter)]

    with Pool(processes=n_workers) as p: 
        p.map(_worker, tiny)
        parts = p.map(_worker, chunks)

    return np.vstack(parts)


def mandelbrot_dask(N, 
                    x_min, x_max, 
                    y_min, y_max, 
                    max_iter=100,
                    n_chunks=32):
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

def benchmark(N=1024, x_min=-2, x_max=1, y_min=-1.5, y_max=1.5, max_iter=100, worker_counts=None, runs=3):
    if worker_counts is None:
        worker_counts=[1,2,4,8]
    
    print("Warming up numba...")
    mandelbrot_serial(64, x_min, x_max, y_min, y_max, max_iter)
    print("Warmed up \n")

    results = {}

    # Serial benchmark
    print("Benchmarking serial")
    times= []

    for _ in range(runs):
        t0 = time.perf_counter()
        mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
        time_run = time.perf_counter() - t0
        times.append(time_run)
    
    median_time = statistics.median(times)
    results["serial"] = median_time
    print(f"Median serial: {median_time:.4f}s", f"(min={min(times):.4f}, max={max(times):.4f})")

    # Parallel benchmark
    times_parallel = {}

    for n in worker_counts:
        print(f"Benchmarking parallel with {n} workers\n")
        times = []
        for _ in range(runs):
            t0=time.perf_counter()
            mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers=n)
            time_run = time.perf_counter() - t0
            times.append(time_run)
        median_time = statistics.median(times)

        times_parallel[n] = median_time

        print(f"Median paralell_{n}: {median_time:.4f}s", f"(min={min(times):.4f}, max={max(times):.4f})")

    results["parallel"] = times_parallel
    
    return results
