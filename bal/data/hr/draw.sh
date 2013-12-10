#! /bin/sh 
gnuplot << EOF 
set output 
plot "./${1}0.dat" using 1:2 title '0' lt rgb "red",  \
     "./${1}1.dat" using 1:2 title '1' lt rgb "blue",  \
     "./${1}2.dat" using 1:2 title '2' lt rgb "green", \
     "./${1}3.dat" using 1:2 title '3' lt rgb "purple"
set term png             
set output "${1}.png" 
replot
set term x11
EOF 
