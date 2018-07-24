#! /bin/bash -eu

echo "# Prerequisite: make deps"
echo "# Prerequisite: make cleanDevEnv"

# move to sloika top-level dir
SLOIKA_ROOT=$(git rev-parse --show-toplevel)

# Create working directory
WORK_DIR=$SLOIKA_ROOT/build/workflow
mkdir -p $WORK_DIR && cd $WORK_DIR


# Fill these in as needed
READ_DIR=/path/to/directory/of/single/read/fast5s
REFERENCE=/path/to/reference/genome.fa
MODEL=/path/to/trained/model.pkl


echo "# 1. Basecall with existing model"

export OMP_NUM_THREADS=1
export THEANO_FLAGS=device=cpu,floatX=float32
$SLOIKA_ROOT/bin/basecall_network.py raw $MODEL $READ_DIR | tee to_map.fa


echo "# 2. Align reads to reference"

# align.py calls BWA to align the basecalls to the reference
bwa index $REFERENCE
$SLOIKA_ROOT/misc/align.py --reference $REFERENCE to_map.fa
# This command extracts a reference sequence for each read using coordinates from the SAM file.
$SLOIKA_ROOT/misc/get_refs_from_sam.py --output_strand_list to_map.txt --pad 50 $REFERENCE to_map.sam | tee to_map_refs.fa


echo "# 3. Remap reads using existing model"

export OMP_NUM_THREADS=1
export THEANO_FLAGS=device=cpu,floatX=float32
$SLOIKA_ROOT/bin/chunkify.py raw_remap --overwrite --input_strand_list to_map.txt --downsample 5 $READ_DIR batch_remapped.hdf5 $MODEL to_map_refs.fa


echo "# 4. Train a new model"

# You may need to adjust these flags for your machine, GPU, and current system load.
#
# Uncomment the following line to train on the GPU:
#export THEANO_FLAGS=openmp=True,floatX=float32,warn_float64=warn,optimizer=fast_run,device=gpu0,lib.cnmem=0.4
TRAIN_DIR=$WORK_DIR/training
$SLOIKA_ROOT/bin/train_network.py raw --overwrite --batch 50 --niteration 1 $SLOIKA_ROOT/models/baseline_raw_gru.py $TRAIN_DIR batch_remapped.hdf5
