Using cub with pytorch.
To build the code, run the following commands from your terminal:
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
$ make
```

```
./cub_test
selected 50
PASS!
```