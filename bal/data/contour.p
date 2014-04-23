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
set xtics ("0.1^8" -8, "0.1^7" -7, "0.1^6" -6, "0.1^5" -5, "0.1^4" -4, "0.001" -3, "0.01" -2, "0.1" -1, "1" 0, "10" 1, "100" 2, "1000" 3, "10^4" 4, "10^5" 5, "10^6" 6, "10^7" 7, "10^8" 8, "10^9" 9) rotate by 270
set ytics ("0.1^8" -8, "0.1^7" -7, "0.1^6" -6, "0.1^5" -5, "0.1^4" -4, "0.001" -3, "0.01" -2, "0.1" -1, "1" 0, "10" 1, "100" 2, "1000" 3, "10^4" 4, "10^5" 5, "10^6" 6, "10^7" 7, "10^8" 8, "10^9" 9) 
#set key font "Times-Roman, 15" 
set terminal pdf
set output outpath

set palette rgbformulae 33,13,10
l '<./contour.sh cont-line.dat 10 25'
p 'cont-iso.dat' with image, 'cont-line.dat' w l lt -1 lw 1.5
