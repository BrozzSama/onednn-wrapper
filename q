#!/bin/bash
#========================================
# Script to submit job in Intel devcloud
#
# Version: 0.5
#========================================
property='i9-10920x'
#property='clx'
#property='skx'
#property='dual_gpu'
if [ -z "$property" ]; then
    property='gpu'
fi

if [ -z "$1" ]; then
	echo "Missing script argument, Usage: ./q run.sh"
elif [ ! -f "$1" ]; then
    echo "File $1 does not exist"
else
	script=$1
	#rm *.sh.* > /dev/null 2>&1
	#qsub
	echo "Submitting job:"
	qsub -l nodes=1:$property:ppn=2 -d . $script -F "$2 $3"
	# qsub -q batch@v-qsvr-nda-l nodes=ppn=2 -I
	# pbsnodes
	#qstat
	qstat 
	#wait for output file to be generated and display
	echo -ne "Waiting for Output."
	until [ -f $script.o* ]; do
		sleep 1
		echo -ne "."
		((timeout++))
		if [ $timeout == 3600 ]; then
			echo "TimeOut 60 seconds: Job is still queued for execution, check for output file later (*.sh.o)"
			break
		fi
	done
	#cat $script.o*
	#cat $script.e*
fi
