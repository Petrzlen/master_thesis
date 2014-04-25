#param inpath 

set xrange [0:1]
set yrange [0:1]
set terminal pdf enhanced font "arial,10" size 5, 4     
set output inpath.".pdf" 

set xlabel "hidden activation on unit 1"
set ylabel "hidden activation on unit 2"

l '<./labels.bash buf.csv 1 3'

#l '<./labels.bash '.inpath.'_0.csv 1 3'
#l '<./labels.bash '.inpath.'_1.csv 1 3'
#l '<./labels.bash '.inpath.'_2.csv 1 3'
#l '<./labels.bash '.inpath.'_3.csv 1 3'

plot inpath."_0.csv" using 1:2 with linespoints notitle lw 0.5 pt 1 lt rgb "red",  \
     inpath."_1.csv" using 1:2 with linespoints notitle lw 0.5 pt 2 lt rgb "blue",  \
     inpath."_2.csv" using 1:2 with linespoints notitle lw 0.5 pt 3 lt rgb "green", \
     inpath."_3.csv" using 1:2 with linespoints notitle lw 0.5 pt 4 lt rgb "purple", \
     inpath."_starts.csv" using 1:2 with points notitle pt 5 lt rgb  "black"

set terminal pngcairo enhanced font "arial,10" size 500, 350 
set output inpath.'.png'
replot
