import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0: return i
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
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N 
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
    mandelbrot_chunk(*args)

def mandelbrot_parallel(N, 
                        x_min, x_max, 
                        y_min, y_max, 
                        max_iter=100, 
                        n_workers=1):
    chunks = []
    n_per_worker= N // n_workers
    for i in range(n_workers):
        row_start = i * n_per_worker
        row_end = N if i == n_workers - 1 else row_start + n_per_worker
        chunks.append((row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter))

    with Pool(processes=n_workers) as pool: 
        out = pool.starmap(mandelbrot_chunk, chunks)


    return np.vstack(out)
        
def benchmark(N=1024, x_min=-2, x_max=1, y_min=-1.5, y_max=1.5, max_iter=100, worker_counts=None, runs=3):
    if worker_counts== None:
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
        speedup = results["serial"] / median_time
        print(f"Median paralell_{n}: {median_time:.4f}s", f"(min={min(times):.4f}, max={max(times):.4f})")

    results["parallel"] = times_parallel
    
    return results

if __name__ == '__main__':
    result=mandelbrot_parallel(4096, x_min=-2, x_max=1.0, y_min=-1.5, y_max=1.5, max_iter=512, n_workers=8)

    normalized = (result / result.max() * 255).astype(np.uint8)
    colored = plt.cm.inferno(normalized, bytes=True)   # returns RGBA uint8 array

    img = Image.fromarray(colored, mode="RGBA")
    out = Path(__file__).parent / 'mandelbrot.png'
    img.save(out)
    # fig, ax = plt.subplots(figsize=(8,6))
    # ax.imshow(result, extent=[-2,1,-1.5,1.5], cmap='inferno', origin='lower', aspect='equal')
    # ax.set_xlabel('Re(c)')
    # ax.set_ylabel('Im(c)')
    # out = Path(__file__).parent / 'mandelbrot.png'
    # fig.savefig(out, dpi=150)

    # results = benchmark(
    #     N=2048, 
    #     max_iter=256, 
    #     worker_counts=[1,2,4,8])

    # print(results)
