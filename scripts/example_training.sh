if [ ! -d "reads" ]
then
    echo "'reads' directory not found"
    exit 1
fi


git clone https://github.com/nanoporetech/sloika
(
	cd sloika
	make cleanDevEnv
)
source sloika/build/env/bin/activate

export OPENBLAS_NUM_THREADS=1
NCPU=`nproc`
THEANO_FLAGS_CPU=device=cpu,floatX=float32,mode=FAST_RUN,blas.ldflags='-lblas',scan.allow_gc=False
THEANO_FLAGS_GPU=device=gpu,floatX=float32,mode=FAST_RUN,blas.ldflags='-lblas',scan.allow_gc=False


#  Generate reference sequences -- replace with your own method
sloika/bin/extract_reference.py reads references.fa

#  Map reads using model -- takes a few hours
THEANO_FLAGS=${THEANO_FLAGS_CPU} sloika/bin/chunkify.py raw_remap --jobs ${NCPU} --chunk_len 4000 --downsample_factor 5 --output_strand_list unfiltered_strands.txt  reads remapped_unfiltered.hdf5 sloika/models/retrained.pkl references.fa

#  Filter reads -- criterion from distribution of mapping scores,  coverage and proportion of stays
( head -n 1 unfiltered_strands.txt ; cat unfiltered_strands.txt | awk '$3 > 0.5 && $3 < 1.2 && ($7 - $6) > 0.95 * $5 && $5 / ($7 - $6 + $5) < 0.55' ) > filtered_strand_list.txt

#  Remap selected reads -- takes a few hours
THEANO_FLAGS=${THEANO_FLAGS_CPU} sloika/bin/chunkify.py raw_remap --jobs ${NCPU} --chunk_len 4000 --downsample_factor 5 --input_strand_list filtered_strand_list.txt --output_strand_list filtered_strands.txt reads remapped_filtered.hdf5 model/baseline.pkl references.fa

#  Train a model
THEANO_FLAGS=${THEANO_FLAGS_GPU} sloika/bin/train_network.py raw  --min_prob 1e-5 sloika/models/raw_1.00_rGr.py training remapped_filtered.hdf5

#  Convert model to CPU
THEANO_FLAGS=${THEANO_FLAGS_GPU} sloika/misc/model_convert.py --target cpu training/model_final.pkl training/model_final_cpu.pkl

#  Basecall (slowly)
THEANO_FLAGS=${THEANO_FLAGS_CPU} sloika/bin/basecall_network.py raw --jobs ${NCPU} training/model_final_cpu.pkl test_reads > basecalls.fa
