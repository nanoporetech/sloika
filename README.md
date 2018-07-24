Sloika is ONT research software for training RNN models for basecalling Oxford Nanopore reads. Sloika is built on top of Theano and is compatible with python 3.4+

## Installation of system prerequisites

    sudo make deps

This will install required system packages on Debian-based Linux distros.

## Setting up clean development environment

    make cleanDevEnv
    source build/env/bin/activate

This will create and activate a python virtual environment in `build/env`.

## Running unit tests in development mode

    make

For this step to function development environment needs to be set up, and make deps must have been installed.

## Note on `THEANO_FLAGS`
To use Theano effectively, A typical set of Theano flags might look like:
```bash
export THEANO_FLAGS=openmp=True,floatX=float32,warn_float64=warn,optimizer=fast_run,device=gpu0,scan.allow_gc=False,lib.cnmem=0.3
```
The Theano flags used for the tests are defined in the `environment` file; you can edit these to test your configuration.

| Flag                | Description |
|---------------------|-------------|
| openmp=True         | Use openmp for calculations. |
| floatX=float32      | Internal floats are single (32bit) precision. This is required for most GPUs. |
| warn_float64=warn   | Warn if double (64bit) precision floats are accidentally used but continue.  warn_float64=raise might be given instead to stop the calculation if a double precision float is encountered.|
| optimizer=fast_run  | Spend more time optimising the expression graph to make the code run faster. For testing optimizer=fast_compile might be used instead. |
| device=gpu0         | Which device to run the calculation on? Common options are cpu and gpuX, where X is the id of the GPU to be used (commonly gpu0). |
| scan.allow_gc=False | Don't allow garbage collection (freeing of memory) during 'scan' operations. This makes recurrent layers quicker at the expensive of higher memory usage. |
| lib.cnmem=0.4       | Use the CUDA CNMEM library for memory allocation. This will improve GPU performance but requires all the memory to be allocated at the beginning of the calculation. The argument is the proportion of the GPU memory to initially allocate.  As a guide, 0.4 is a good number for training since it allows two runs to both use the same GPU. For programs run on a per-read basis, basecalling and mapping, a smaller proportion like 0.05 is more appropriate. |

 
