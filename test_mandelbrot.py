from mandelbrot import mandelbrot_naive, mandelbrot_naive_numba
from mandelbrot_parallel import mandelbrot_pixel, mandelbrot_pixel_numba, mandelbrot_chunk, mandelbrot_serial, mandelbrot_parallel, mandelbrot_dask
from dask.distributed import Client
import pytest
import numpy as np

args = (-2.0, 1.0, -1.5, 1.5)

# WIDTH, HEIGHT = 1024, 1024
MAX_ITER = 20

# ▶ Numba @njit:
#   ▶ The compiled function is behaviourally identical to the Python version
#   ▶ Test the Python version first; the Numba version passes the same tests
#   ▶ Warm-up call needed before timing, not before correctness tests

KNOWN_CASES = [
    (0+0j, 100, 100),  # origin: never escapes
    (5.0+0j, 100, 1),  # far outside, escapes on iteration 1
    (-2.5+0j, 100, 1), # left tip of set
]
IMPLEMENTATIONS = [mandelbrot_pixel, mandelbrot_pixel_numba]

@pytest.mark.parametrize("impl", IMPLEMENTATIONS) # independent axis
@pytest.mark.parametrize("c, max_iter, expected", KNOWN_CASES) # bundled
def test_pixel_all(impl, c, max_iter, expected):
    assert impl(c.real, c.imag, max_iter) == expected # 2 x 3 = 6 tests

def test_naive_equals_numba():
    result_naive = mandelbrot_naive(*args, max_iter=MAX_ITER)

    result_naive_numba = mandelbrot_naive_numba(*args, max_iter=MAX_ITER)
    # Assert that the array from naive is equal to array from naive_numba
    assert np.array_equal(result_naive, result_naive_numba)

# ▶ Multiprocessing Pool:
#   ▶ Test the worker function (mandelbrot pixel) in isolation — it is pure
#   ▶ Integration test: the assembled grid must match the serial result on a small grid
#   ▶ Do not test Pool machinery — that is Python’s responsibility

def test_mandelbrot_pixel_basic():
    # inside the mandelbrot set.
    assert mandelbrot_pixel_numba(0.0, 0.0, 100) == 100

    # outside the mandelbrot set
    assert mandelbrot_pixel_numba(2.0, 2.0, 100) < 5

    # boundary point
    assert mandelbrot_pixel_numba(-0.73, 0.236, 100) > 10

def test_mandelbrot_chunk():
    N = 4
    res = mandelbrot_chunk(0, N, N, *args, 20)

    assert res.shape == (4, 4)
    assert res.dtype == np.int32

    assert res[0,0] >= 0
    assert res[3,3] >= 0

def test_parallel_equals_serial():
    N = 32

    serial = mandelbrot_serial(N, *args, max_iter=MAX_ITER)
    parallel = mandelbrot_parallel(N, *args, max_iter=MAX_ITER, n_workers=2)

    assert np.array_equal(serial, parallel)

def test_chunk_sizes(): 
    N = 64 

    res1 = mandelbrot_parallel(N, *args, max_iter=MAX_ITER, n_workers=2, n_chunks=2)
    res2 = mandelbrot_parallel(N, *args, max_iter=MAX_ITER, n_workers=2, n_chunks=8)

    assert np.array_equal(res1, res2)

# ▶ Dask:
#   ▶ Test the underlying compute function, not the scheduler
#   ▶ Integration: future = client.submit(f, arg); assert client.gather(future) == expected

def test_dask_equals_serial():
    N = 64

    serial = mandelbrot_serial(N, *args, max_iter=MAX_ITER)
    dask = mandelbrot_dask(N, *args, max_iter=MAX_ITER)

    assert np.array_equal(serial, dask)

def test_dask_submit_equals_serial():
    client = Client(processes=False)

    N = 64

    serial = mandelbrot_serial(N, *args, max_iter=MAX_ITER)

    future = client.submit(mandelbrot_chunk, 0, N, N, *args, MAX_ITER)

    result = client.gather(future)

    assert np.array_equal(serial, result)

    client.close()

def test_dask_chunked():
    client = Client(processes=False)

    N = 64 
    
    serial = mandelbrot_serial(N, *args, max_iter=MAX_ITER)

    futures = []
    chunk_size = 8

    for row_start in range(0, N, chunk_size):
        row_end = min(row_start + chunk_size, N)

        futures.append(
            client.submit(mandelbrot_chunk, 
                          row_start, row_end, N, 
                          *args, MAX_ITER)
                          )
    
    parts = client.gather(futures)
    result = np.vstack(parts)

    assert np.array_equal(serial, result)

    client.close()

def test_cross_validation_small_grid():
    N = 32

    serial_res = mandelbrot_serial(N, *args, max_iter=MAX_ITER)

    numba_res = mandelbrot_chunk(0, N, N, *args, max_iter=MAX_ITER)
    parallel_res = mandelbrot_parallel(N, *args, max_iter=MAX_ITER, n_workers=2)
    dask_res = mandelbrot_dask(N, *args, max_iter=MAX_ITER)

    assert np.array_equal(numba_res, serial_res)
    assert np.array_equal(parallel_res, serial_res)
    assert np.array_equal(dask_res, serial_res)

