#!/bin/bash

OUTDIR="./perf"
OUTFILE="$OUTDIR/measures-rowcol.txt"
BASEFILE="$OUTDIR/baseline-rowcol.txt"


gcc -o perfs -fopenmp ./perftest.c

for dim in 256 512 1024 2048
do
	echo "Evaluating baseline (serial code)..."
	./perfs $dim 1 >> $BASEFILE

	for n in 4 8 16 32 64
	do
		echo "Testing $dim x $dim matrices with concurrency level $n... "
		./perfs $dim $n >> $OUTFILE
	done	
done
