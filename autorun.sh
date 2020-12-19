#!/bin/bash

binary=$1
N1=$2
N2=$3
STEPS=20
delta=$(((N2-N1)/${STEPS}))

for i in $(seq 0 $STEPS)
do
	./${binary} $((N1+$i*$delta))
done