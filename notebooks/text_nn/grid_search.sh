#!/usr/bin/env bash
for cont_lam in 0 .0001 .001
do
    for sel_lam in .001 .0025 .005 .0075 .01 .025 .05 .075 .1
    do
        sbatch runner.sh --export=cont_lam=${cont_lam},sel_lam=${sel_lam}
    done
done