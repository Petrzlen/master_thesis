#!/bin/bash 

echo "preparing progress of $1 (label $2) and output to $3" 

bash performance.sh 'auto4_bal_orig_best' 100 0.2
mv $1.dat sim1.dat
bash performance.sh 'auto4_tlrbbest' 100 0.2
mv $1.dat sim2.dat
bash performance.sh 'auto4_tlr_best_bcan' 100 0.2
mv $1.dat sim3.dat

#params col
#params sim1 sim2 
#params col_lab, sim1_lab, sim2_lab
gnuplot -e "outpath='$3';col_lab='$2';" epoch_cmp.p



