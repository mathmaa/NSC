import cProfile, pstats
from mandelbrot import mandelbrot_numpy

mandelbrot_numpy()
cProfile.run('mandelbrot_numpy()', 'numpy_profile.prof')

stats  = pstats.Stats('numpy_profile.prof')
stats.sort_stats('cumulative')
stats.print_stats()