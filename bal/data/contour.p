#http://www.dommelen.net/l2h/contour.txt
#http://gnuplot-tricks.blogspot.sk/2009/07/maps-contour-plots-with-labels.html

reset
set xtic auto  
set ytic auto  
set isosample 50, 50
set dgrid3d 50, 50
set table 'cont-iso.dat'
splot inpath
unset table

set contour base
set cntrparam level incremental 0, 5, 100
#set cntrparam level incremental 2.5, 0.25, 5.5
unset surface
set table 'cont-line.dat'
splot inpath
unset table

reset
unset key
set xtic auto  
set ytic auto  
set terminal pdf
set output outpath

set palette rgbformulae 33,13,10
l '<./contour.sh cont-line.dat 0 5 0'
p 'cont-iso.dat' with image, 'cont-line.dat' w l lt -1 lw 1.5
