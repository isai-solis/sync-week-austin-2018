# sync-week-austin-2018
Sync Week Project for Austin team 2018

# objective
The goal of this project is to build a pair of tools that will allow one to:

1. Identify people in a given video stream.
2. Blur-out the faces present in a give video stream.

# Build and Test
Once all the dependencies are installed, you can test the code by typing:

```
python -m unittest discover
```

or a single test:

```
python -m unittest test.test_pixelator
```

# darknet/YOLO
For consistency, you should download YOLO to the base of this project, e.g.

```
git clone https://github.com/pjreddie/darknet darknet
```

This way, after compiled artifacts are available via something like this:

```
from ctypes import *
import os

lib = CDLL(os.path.join(path_to_project_base, 'darknet/libdarknet.so'), RTLD_GLOBAL)
```
