Release 2.0
===========

* Migrate to python3 in full and stop supporting python2
* Open-source project


Release 1.2 brown bag 1
=======================

* Bug fixes to chunkify from raw data
* Take the `--drop` into account when reporting label accuracy during training


Release 1.2
===========

* Can train and basecall off raw data
  * Examples in `models/` directory use temporal convolution to extract features from the raw signal
  * Training data may be labelled from events or remapped on the fly using a pretrained raw data model
  * The interfaces for most scripts have changed to support raw data:
    * `./bin/basecall_network` and `./bin/train_network.py` provide `raw` and `events` routes
    * `./bin/chunkify.py` provides `identity`, `remap`, `raw_identity` and `raw_remap` routes
* `./bin/basecall_network.py` may be used via `./bin/basecall_network` entry point that sets up its environment
* Full builds are performed on CI only for `master`, `release` and branches with names having `ci_` prefix
* Layers:
  * All layers now have `insize` and `size` attributes; models pickled with previous versions of Sloika will not be unpicklabe in Sloika 1.2
  * Added `Convolution` and `MaxPool` layers
* Improvements to `pekarnya.py`:
  * Updated database schema
  * Runs are marked with time stamps and git commits
  * Uniquely generated output directories facilitate restart of failed jobs
* New script `./bin/extract_reference.py` for extraction of references from a directory of fast5 files


### Minor changes

* Minimum required numpy version bumped to 1.9.0
* `verify_network.py` tests network execution on random inputs
* `align.py` outputs reference coordinates
* `align.py` fixed under python3
* Randomly choosing chunk size during training is now the default


Release 1.1 brown bag 3
=======================

Addresses the following issue:
* Make Sloika 1.1 avoid h5py version 2.7.0. This version works in Dev environment, but we are unable to import it on CI nodes
    https://github.com/h5py/h5py/issues/803


Release 1.1 brown bag 2
=======================

Addresses the following issues:
* Brown bag 1 did not update changelog
    https://git.oxfordnanolabs.local/algorithm/sloika/issues/43
* Fixes TypeError in json dump script when invoked with `--out_file` option
    https://git.oxfordnanolabs.local/algorithm/sloika/issues/52


Release 1.1 brown bag 1
=======================

Addresses the following issues:
* Remapping doesn't appear to work with 2d reads
    https://git.oxfordnanolabs.local/algorithm/sloika/issues/40
* Can't change segmentation in chunkify remap
    https://git.oxfordnanolabs.local/algorithm/sloika/issues/41
* Work around Isilon problems
    https://git.oxfordnanolabs.local/algorithm/sloika/issues/42


Release 1.1
===========

* Activation functions have been separated into their own module and many new functions have been added
    See https://wiki/display/~tmassingham/2016/10/17/Activation+functions
    Note: this rearrangement breaks compatibility with older model pickle files
* Refactoring of `NBASE` constant
    Now a single source of responsibility `sloika/variables.py`
    Models importing `_NBASE` from `sloika/module_tools.py` should now import `NBASE` instead
* Default for training and basecalling are transducer based models
* Compilation of networks is handled automatically by `basecall_network.py`
  * Compiled network may be saved for future use
  * `compile_network.py` executable has been removed
* Recurrent layers
  * New recurrent unit types have been added
  * Detailed tests to ensure recurrent layers work
  * Type of gate function is now an option on layer initialisation
* Pekarnya server for scheduling model training jobs
    https://wiki/display/RES/Pekarnya
* Considerable work on the building and testing infrastructure
  * Stable and development branches were created
  * Binary artefacts are built for each commit in development branch
  * Artefacts are automatically versioned in development branch
  * Unit and acceptance tests are exercising artefact before it is marked as a release candidate
* Remapping using RNN from fast5 directly to chunks
  * `chunkify.py`
    * `chunkify.py identity` has similar behaviour to `chunk_hdf5.py`
    * `chunkify.py remap` will remap a directory of fast5 files using a transducer RNN before chunking
  * `remap_hdf5.py` and `chunk_hdf5.py` removed in favour of `chunkify.py`
  * Per chunk normalisation optional `--normalisation`
    * Default is still to normalise over entire read
* Chunk size for training can be randomly selected from batch to batch
  * `--chunk_len_range min max`
  * Default is to always train with maximum possible chunk size
  * Chunk size chosen uniformly in specified interval
* Edge events are not used when assessing loss function
  * `--drop n`
  * Default to drop 20 events from start and end before assessing loss


### Minor changes

* Changed default trimming from ends of sequence
* Fix to allow trimming of zero events
* Minimum read length (in events) for chunking to take place
* Removed vestigial `networks.py` file that has been replaced by the contents of the `models/` directory
* Seed for random number generator can be set on command line of `train_network.py`
* Enable HDF5 compression
* Fix to ensure every chunk starts with a non-zero (not stay) label
* Trim first and last events from loss function calculation (burn-in)
* Fix bug in how kmers are merged into sequence in low complexity regions
* Increased PEP8 compliance
* Default location of segmentation information has changed (see Untangled 0.5.1)
* Location of segmentation information can now be given as commandline option in many programs
* Trainer copies logging information to stdout.  May be silenced with `--quiet`
* JSON may be dumped to file rather than stdout


Release 1.0
===========

Initial release
