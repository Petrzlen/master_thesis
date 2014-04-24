#!/bin/bash

gnuplot -e "inpath='stats/auto4_bal_orig_back/log_lls_0.dat';outpath='../../text/img/tlr-auto4-success.pdf';val_from=0;val_d=5;val_to=100;" contour.p

awk '{if(NF==2) print $1,$2,50000; else print $0}' stats/auto4_bal_orig_back/log_lle_0.dat | awk '{print $1,$2,($3>10000)?10000:$3}' > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/tlr-auto4-epoch.pdf';val_from=0;val_d=-1000;val_to=10000;" contour.p

