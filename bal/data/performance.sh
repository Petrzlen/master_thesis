#!/bin/bash 

echo "performance files $1 preprocessing with min_count=$2 and log step=$3" 

p="{if(NR>1&&\$2>=$2)print \$0;}"
s="BEGIN{ls=0;}{if(log(\$1+1)>=ls) {ls = log(\$1+1)+$3; print \$1,\$2,\$3,\$4,\$5,\$6,\$7;}}END{print}" 
echo $s 

#could make +- 2 promile
awk "$p" stats/$1/epoch_bs_f.dat | awk "$s" > tbsf.dat
awk "$p" stats/$1/epoch_bs_b.dat | awk "$s" > tbsb.dat
awk "$p" stats/$1/epoch_ps_f.dat | awk "$s" > tpsf.dat
awk "$p" stats/$1/epoch_ps_b.dat | awk "$s" > tpsb.dat
