#!/usr/bin/env bash
for sel_lam in 0 .0001 .001
do
    for cont_lam in .01 .025 .05 .075 .1 .25 .5
    do
        cat  runner.sh.template | sed -e "s/sellam/${sel_lam}/g" -e "s/contlam/${cont_lam}/g" > runner.sh
        sbatch runner.sh
    done
done