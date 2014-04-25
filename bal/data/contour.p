#Lessons learned: if gnuplot is plotting weird, check your data
#  if they are ok, check them again
#  and again 

#http://www.dommelen.net/l2h/contour.txt
#http://gnuplot-tricks.blogspot.sk/2009/07/maps-contour-plots-with-labels.html

#params inpath, outpath (files) 
#params val_from, val_d, val_to (contour) 
#params rxf,rxt, ryf,ryt (ranges) 
#params rgb_a, rgb_b, rbc_c 

reset
#set xtic auto  
#set ytic auto  

#http://stackoverflow.com/questions/5864670/3d-mapped-graph-with-gnuplot-not-accurate
#https://groups.google.com/forum/#!topic/comp.graphics.apps.gnuplot/Eh7F2Wk3zDk

#set isosample 29, 24
#set dgrid3d 

#TODO logscale 
set xrange [rxf:rxt]
set yrange [ryf:ryt]

set table 'cont-iso.dat'
splot inpath
unset table

set contour base
set cntrparam level incremental val_from, val_d, val_to
unset surface
set table 'cont-line.dat'
splot inpath
unset table

reset
unset key
set xrange [rxf:rxt]
set yrange [ryf:ryt]

set xtics ("0.1^9" -9, "0.1^8" -8, "0.1^7" -7, "0.1^6" -6, "0.1^5" -5, "0.1^4" -4, "0.001" -3, "0.01" -2, "0.1" -1, "1" 0, "10" 1, "100" 2, "1000" 3, "10^4" 4, "10^5" 5, "10^6" 6, "10^7" 7, "10^8" 8, "10^9" 9) rotate by 270
set ytics ("0.1^9" -9, "0.1^8" -8, "0.1^7" -7, "0.1^6" -6, "0.1^5" -5, "0.1^4" -4, "0.001" -3, "0.01" -2, "0.1" -1, "1" 0, "10" 1, "100" 2, "1000" 3, "10^4" 4, "10^5" 5, "10^6" 6, "10^7" 7, "10^8" 8, "10^9" 9) 
#set key font "Times-Roman, 15" 
set terminal pdf font "arial,8" #size 500, 350 
set output outpath

#http://stackoverflow.com/questions/19294342/heatmap-with-gnuplot-on-a-non-uniform-gridless

#set pm3d interpolate 4,4
set palette rgbformulae rgb_a,rgb_b,rgb_c
l '<./contour.sh cont-line.dat 10 20 7'
p inpath u 1:2:3 with image, 'cont-line.dat' w l lt -1 lw 1.5

#set terminal pngcairo enhanced font "arial,10" size 800, 800 
#set output "motac.png" 
#replot
