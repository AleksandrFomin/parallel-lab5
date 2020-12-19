#!/bin/bash

LOG_FILE=./cpu.log

>${LOG_FILE}

while [ 1 ]
do
	TIME=$(date "+%Y-%m-%d %H:%M:%S.%N")
	CPU_LOAD=$(head -5 /proc/stat)
	printf "%s\n%s\n" "${TIME}" "${CPU_LOAD}" >> ${LOG_FILE}
	sleep 0.2
done