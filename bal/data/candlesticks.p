#http://gnuplot.sourceforge.net/demo/candlesticks.html
# set terminal pngcairo  transparent enhanced font "arial,10" fontscale 1.0 size 500, 350 
# set output 'candlesticks.6.png'
set boxwidth 0.1 absolute

set xrange [:]
set yrange [:]
plot 'candlesticks.dat' using 1:3:2:6:5 with candlesticks lt 3 lw 2 title 'Quartiles' whiskerbars, '' using 1:4:4:4:4 with candlesticks lt -1 lw 2 notitle

