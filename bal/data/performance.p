#!/usr/bin/env gnuplot

#params outpath 

#set boxwidth 0.2 absolute
set bars 1.0 
set xrange [:] noreverse nowriteback
set yrange [:] noreverse nowriteback

set key outside;
set key right top;
set key box lw 2; 

#fake logarithm plot 
#set xtics ("0" 200, "100" 300, "1000" 1200, "10000" 10200, "100000" 100200, "1000000" 1000200) 

set ytics ("1" 1, "0.8" 0.8, "0.6" 0.6, "0.4" 0.4, "0.2" 0.2, "0" 0) 

set xlabel "epoch"
set ylabel "network performance"

set logscale x 

set grid ytics lw 5 lc rgb "#dddddd"
set grid xtics lw 5 lc rgb "#dddddd"

#lc rgb "#880088"
set terminal unknown
plot 'tpsb.dat' u 1:5:($5-$6):($5+$6) with errorlines \
        lt 3 lw 3 pt 5 ps 0.5 title 'patSuccB', \
     'tpsf.dat' u 1:5:($5-$6):($5+$6) with errorlines \
        lt 1 lw 3 pt 5 ps 0.5 title 'patSuccF', \
     'tbsb.dat' u 1:(1-$5):(1-$5-$6):(1-$5+$6) with errorlines \
        lt 4 lw 2 pt 5 ps 0.5 title 'bitErrB', \
     'tbsf.dat' u 1:(1-$5):(1-$5-$6):(1-$5+$6) with errorlines \
        lt 7 lw 2 pt 5 ps 0.5 title 'bitErrF'

set terminal pdf font "arial,8" size 7, 3
#set xrange [GPVAL_DATA_X_MIN-50:] noreverse nowriteback
set output outpath
replot 
