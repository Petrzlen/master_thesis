#! /bin/sh 
gnuplot << EOF 
set xrange [0:1]
set yrange [0:1]
plot "./${1}0.dat" using 1:2 notitle lt rgb "red",  \
     "./${1}1.dat" using 1:2 notitle lt rgb "blue",  \
     "./${1}2.dat" using 1:2 notitle lt rgb "green", \
     "./${1}3.dat" using 1:2 notitle lt rgb "purple"
set term png             
set output "${1}.png" 
replot
set term x11
