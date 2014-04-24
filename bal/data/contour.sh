#!/bin/bash
#http://gnuplot-tricks.blogspot.sk/2009/07/maps-contour-plots-with-labels.html

gawk -v k=$2 -v m=$3 -v p=$4 'function abs(x) { return (x>=0?x:-x) }
    BEGIN{level=k;}
    {
            if($0~/# Contour/) {nr=0; level = (level + p) % m;}
            if(NF<=0) nr=0
            if(nr % m == level) {a[i]=$1; b[i]=$2; c[i]=$3;}
            if(nr % m == (level+m-1)%m) {i++; x = $1; y = $2;}
            if(nr % m == (level+m+1)%m) r[i]= 180.0*atan2(y-$2, x-$1)/3.14
            nr++
    }
    END {   if(d==0) {
                    for(j=1;j<=i;j++)
                    printf "set label %d \"%g\" at %g, %g centre front rotate by %d\n", j, c[j], a[j], b[j], r[j]
            }
    }' $1
