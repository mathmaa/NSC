import cProfile, pstats
from mandelbrot import mandelbrot_naive, mandelbrot_numpy

mandelbrot_naive()
mandelbrot_numpy()
cProfile.run('mandelbrot_naive()', 'naive_profile.prof')
cProfile.run('mandelbrot_numpy()', 'numpy_profile.prof')

for name in ('naive_profile.prof','numpy_profile.prof'):    
    stats  = pstats.Stats(name)
    stats.sort_stats('cumulative')
    stats.print_stats(10)