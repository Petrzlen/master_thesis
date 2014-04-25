#!/bin/bash

sh performance.sh 'auto4_tlr_best' 500 0.2
gnuplot -e "outpath='../../text/img/tlr-best-perf.pdf'" performance.p

#sh performance.sh 'auto4_tlr_best_can' 500 0.2
#gnuplot -e "outpath='../../text/img/tlr-best-can.pdf'" performance.p

#======================= MOMENTUM ==================
declare -a mom_arr=("0.001" "0.003" "0.01" "0.03" "0.1" "0.3")

## now loop through the above array
for mom in "${mom_arr[@]}"
do
  suf=`echo $mom | sed 's/\./-/g'`
  echo "tlr-mom-auto4-success-$suf.pdf"
  less "stats/auto4_tlr_mom/lls_0.dat" | awk "{if(\$3==$mom) print \$1,\$2,\$4;}" | bash to_log_grid.bash > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/tlr-mom-auto4-success-$suf.pdf';val_from=0;val_d=5;val_to=100;rxf=-3;rxt=8;ryf=-9;ryt=2;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "tlr-mom-auto4-epoch-$suf.pdf"
  less "stats/auto4_tlr_mom/lle_0.dat" | awk "{if(\$3==$mom) print \$1,\$2,\$4;}" | bash to_log_grid.bash | bash post_epochs.bash 20000 > "buf.dat"
gnuplot -e "inpath='buf.dat';outpath='../../text/img/tlr-mom-auto4-epoch-$suf.pdf';val_from=0;val_d=1000;val_to=5000;rxf=-3;rxt=8;ryf=-9;ryt=2;rgb_a=10;rgb_b=13;rgb_c=33" contour.p
done

#ls | grep -o 'k3_4_[7-9]_139[0-9]\+' | sort | uniq | while read filename
#do
#  echo "motam $filename"
#  bash zmotaj_stats.bash $filename $filename
#done  

#=================== TLR ====================
  #=================== auto4 ====================
  echo "tlr-auto4-success.pdf"
gnuplot -e "inpath='stats/auto4_bal_orig/log_lls_0.dat';outpath='../../text/img/tlr-auto4-success.pdf';val_from=0;val_d=5;val_to=100;rxf=-4;rxt=9;ryf=-9;ryt=1;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "tlr-auto4-epoch.pdf"
  less stats/auto4_bal_orig/log_lle_0.dat | bash post_epochs.bash 20000 > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/tlr-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=5000;rxf=-4;rxt=9;ryf=-9;ryt=1;rgb_a=10;rgb_b=13;rgb_c=33" contour.p

#=================== BAL RECIRC ====================
  #=================== auto4 ====================
#awk '{if(NF<=1||(-3.5<$1 && $1<2)) print $0}' stats/auto4_bal_recirc/log_lls_0.dat > buf.dat

  echo "bal-recirc-auto4-success.pdf"
gnuplot -e "inpath='stats/auto4_bal_recirc/log_lls_0.dat';outpath='../../text/img/bal-recirc-auto4-success.pdf';val_from=0;val_d=5;val_to=100;rxf=-4;rxt=2;ryf=-9;ryt=2;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "bal-recirc-auto4-epoch.pdf"
  less stats/auto4_bal_recirc/log_lle_0.dat | bash post_epochs.bash 20000 > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/bal-recirc-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=50000;rxf=-4;rxt=2;ryf=-9;ryt=2;rgb_a=10;rgb_b=13;rgb_c=33" contour.p

#=================== BAL RECIRC ====================
  #=================== auto4 ====================
#awk '{if(NF<=1||(-3.5<$1 && $1<2)) print $0}' stats/auto4_bal_recirc/log_lls_0.dat > buf.dat

  echo "generec-auto4-success.pdf"
gnuplot -e "inpath='stats/auto4_generec/log_lls_0.dat';outpath='../../text/img/generec-auto4-success.pdf';val_from=0;val_d=5;val_to=100;rxf=-4;rxt=2;ryf=-7;ryt=2;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "generec-auto4-epoch.pdf"
less stats/auto4_generec/log_lle_0.dat | bash post_epochs.bash 20000 > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/generec-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=50000;rxf=-4;rxt=2;ryf=-7;ryt=2;rgb_a=10;rgb_b=13;rgb_c=33" contour.p

#=================== auto4 ====================
ls | grep -o 'k3_139[0-9]\+_[3-9]' | sort | uniq | while read filename
do
  num=$(echo $filename | grep -o '[0-9]$')
  echo "motam $filename with num=$num"
  
  #bash zmotaj_stats.bash w $filename w 
  
  echo "tlr-k3-$num-success.pdf"
  gnuplot -e "inpath='stats/$filename/log_lls_0.dat';outpath='../../text/img/k3/tlr-$num-success.pdf';val_from=0;val_d=10;val_to=100;rxf=-4;rxt=4;ryf=-7;ryt=1;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "tlr-k3-$num-epoch.pdf"
  less stats/$filename/log_lle_0.dat | bash post_epochs.bash 5000 > buf.dat
  gnuplot -e "inpath='buf.dat';outpath='../../text/img/k3/tlr-$num-epoch.pdf';val_from=0;val_d=300;val_to=2500;rxf=-4;rxt=4;ryf=-7;ryt=1;rgb_a=10;rgb_b=13;rgb_c=33" contour.p
done
  

