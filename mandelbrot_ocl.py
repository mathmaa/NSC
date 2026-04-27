import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt

X_MIN, X_MAX = -2.0, 1.0
Y_MIN, Y_MAX = -1.5, 1.5
MAX_ITER = 100

KERNEL_F32 = """
__kernel void mandelbrot(
	__global int *out,
	const float x_min, const float x_max,
	const float y_min, const float y_max,
	const int N, const int max_iter)
{
	int col = get_global_id(0);
	int row = get_global_id(1);

	if (col >= N || row >= N) return;

	float c_real = x_min + col * (x_max - x_min) / (float)N;
	float c_imag = y_min + row * (y_max - y_min) / (float)N;

	float zr = 0.0f;
	float zi = 0.0f;
	int count = 0; 

	while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
		float tmp = zr*zr - zi*zi + c_real;
		zi = 2.0f * zr * zi + c_imag;
		zr = tmp;
		count++;
	}

	out[row * N + col] = count;
}
"""

KERNEL_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot(
	__global int *out,
	const double x_min, const double x_max,
	const double y_min, const double y_max,
	const int N, const int max_iter)
{
	int col = get_global_id(0);
	int row = get_global_id(1);

	if (col >= N || row >= N) return;

	double c_real = x_min + col * (x_max - x_min) / (double)N;
	double c_imag = y_min + row * (y_max - y_min) / (double)N;

	double zr = 0.0;
	double zi = 0.0;
	int count = 0; 

	while (count < max_iter && zr*zr + zi*zi <= 4.0) {
		double tmp = zr*zr - zi*zi + c_real;
		zi = 2.0 * zr * zi + c_imag;
		zr = tmp;
		count++;
	}

	out[row * N + col] = count;
}
"""

def supports_fp64(ctx):
	return any("cl_khr_fp64" in dev.extensions for dev in ctx.devices)

def mandelbrot_gpu(
	ctx, queue, N,
	dtype=np.float32,
	x_min=X_MIN, x_max=X_MAX, 
	y_min=Y_MIN, y_max=Y_MAX,
	max_iter=MAX_ITER
):
	if dtype == np.float32:
		prog = cl.Program(ctx, KERNEL_F32).build()
		cast_float = np.float32
	elif dtype == np.float64:
		prog = cl.Program(ctx, KERNEL_F64).build()
		cast_float = np.float64
	else:
		raise ValueError("dtype must be np.float32 or np.float64")
	
	kernel = cl.Kernel(prog, "mandelbrot")

	out = np.zeros((N,N), dtype=np.int32)
	out_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)

	kernel(
		queue, (N, N), None, 
		out_dev,
		cast_float(x_min), cast_float(x_max),
		cast_float(y_min), cast_float(y_max),
		np.int32(N), np.int32(max_iter),
	)

	cl.enqueue_copy(queue, out, out_dev)
	queue.finish()

	return out


def benchmark(func, *args, runs=3):
	import statistics
	times = []
	for _ in range(runs):
		t0 = time.perf_counter()
		func(*args)
		times.append(time.perf_counter() - t0)
	return statistics.median(times)

if __name__ == "__main__":
	N = 8192
	RUNS = 3
	
	ctx = cl.create_some_context(interactive=False)
	queue = cl.CommandQueue(ctx)
	dev = ctx.devices[0]
	
	print(f'Device: {dev.name}\n')

	_ = mandelbrot_gpu(ctx, queue, N=64)

	print(f"Benchmarking N={N}, max_iter={MAX_ITER}, {RUNS} runs each:\n")
	t32 = benchmark(mandelbrot_gpu, ctx, queue, N, np.float32, runs=RUNS)
	img_f32 = mandelbrot_gpu(ctx, queue, N, np.float32)
	print(f"  float32: {t32*1e3:.1f} ms")

	img_f64 = None
	t64 = None

	if supports_fp64(ctx):
		img_f64 = mandelbrot_gpu(ctx, queue, N, np.float64)
		t64 = benchmark(mandelbrot_gpu, ctx, queue, N, np.float64, runs=RUNS)
		print(f"  float64: {t64*1e3:.1f} ms")
		print(f"  Ratio float64/float32: {t64/t32:.2f}x")

		diff = np.abs(img_f32.astype(int) - img_f64.astype(int))
		print(f"  Max pixel difference (f32 vs f64): {diff.max()}")
	else:
		print("  float64: not supported on this device")
	
	fig, axes = plt.subplots(1, 2 if img_f64 is not None else 1,
							 figsize=(12 if img_f64 is not None else 6, 5))
	if img_f64 is None:
		axes = [axes]

	axes[0].imshow(img_f32, cmap='hot', origin='lower')
	axes[0].set_title(f"float32  ({t32*1e3:.1f} ms)")
	axes[0].axis('off')

	if img_f64 is not None:
		axes[1].imshow(img_f64, cmap='hot', origin='lower')
		axes[1].set_title(f"float64  ({t64*1e3:.1f} ms)")
		axes[1].axis('off')

	plt.suptitle(f"GPU Mandelbrot  N={N}  device: {dev.name}", fontsize=10)
	plt.tight_layout()
	out = "mandelbrot_opencl.png"
	plt.savefig(out, dpi=150)
	plt.show()
	print(f"\nSaved to {out}")
