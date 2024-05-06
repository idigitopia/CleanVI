map_size_x=25
map_size_y=25

python benchmark.py --map_size $map_size_x $map_size_y --headless --mode python
python benchmark.py --map_size $map_size_x $map_size_y --headless --mode numpy
python benchmark.py --map_size $map_size_x $map_size_y --headless --mode torch_cpu
# python benchmark.py --map_size $map_size_x $map_size_y --headless --mode torch_gpu
python benchmark.py --map_size $map_size_x $map_size_y --headless --mode mlx