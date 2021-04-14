Using cub with pytorch. Need CUDA 11.x to be able to work.

To build the code, run the following commands from your terminal:

```bash
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
$ make
```

```bash
./cub_test
selected 50
PASS!
```
