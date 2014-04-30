#!/usr/bin/env gnuplot
#compares two simulation statistics [col to epoch] 

#params col_lab
#params outpath 

set bars 1.0 
set xrange [1:100001] noreverse nowriteback
set yrange [:] noreverse nowriteback

set key outside;
set key right top;
set key box lw 2; 

set xlabel "epoch"  
set ylabel col_lab

set logscale x 

set grid ytics lw 5 lc rgb "#dddddd"
set grid xtics lw 5 lc rgb "#dddddd"

set terminal pdf font "arial,8" size 6, 3
set output outpath
plot 'sim1.dat' u ($1+1):5:($5-$6):($5+$6) with errorlines \
        lt 3 lw 3 pt 5 ps 0.5 title 'BAL', \
     'sim2.dat' u ($1+1):5:($5-$6):($5+$6) with errorlines \
        lt 1 lw 3 pt 5 ps 0.5 title 'TLR', \
     'sim3.dat' u ($1+1):5:($5-$6):($5+$6) with errorlines \
        lt 4 lw 3 pt 5 ps 0.5 title 'TLR-can' 

