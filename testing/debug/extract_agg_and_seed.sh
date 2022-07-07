#!/bin/bash

FILES="runs/*"
mkdir simple_runs
mkdir simple_runs/agg/
mkdir simple_runs/seed0/
for f in $FILES
do
	FILESTWO=$f/*
	for ft in $FILESTWO
	do
		agg="$ft/aggregate/events*"
		agg=( $agg )
		agg=${agg[0]}
		zero="$ft/seed=0/events*"
		zero=( $zero )
		zero=${zero[0]}
		
		ft_under=${ft//[\/]/_}
		ft_under=${ft_under//runs_/}

		fn=$(basename $agg)
		
		mkdir simple_runs/agg/$ft_under
		mkdir simple_runs/seed0/$ft_under

		cp $agg simple_runs/agg/$ft_under/$fn
		cp $zero simple_runs/seed0/$ft_under/$fn

	done
done
