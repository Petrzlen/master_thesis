#!/bin/bash

sh performance.sh 'auto4_tlr_best' 500 0.2
gnuplot -e "outpath='../../text/img/tlr-best-perf.pdf'" performance.p

exit 

#=================== TLR ====================
  #=================== auto4 ====================
  echo "tlr-auto4-success.pdf"
gnuplot -e "inpath='stats/auto4_bal_orig/log_lls_0.dat';outpath='../../text/img/tlr-auto4-success.pdf';val_from=0;val_d=5;val_to=100;rxf=-4;rxt=9;ryf=-9;ryt=1;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "tlr-auto4-epoch.pdf"
awk '{if(NF==2) print $1,$2,50000; else print $0}' stats/auto4_bal_orig/log_lle_0.dat | awk '{print $1,$2,($3>10000)?10000:$3}' > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/tlr-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=10000;rxf=-4;rxt=9;ryf=-9;ryt=1;rgb_a=10;rgb_b=13;rgb_c=33" contour.p

#=================== BAL RECIRC ====================
  #=================== auto4 ====================
#awk '{if(NF<=1||(-3.5<$1 && $1<2)) print $0}' stats/auto4_bal_recirc/log_lls_0.dat > buf.dat

  echo "bal-recirc-auto4-success.pdf"
gnuplot -e "inpath='stats/auto4_bal_recirc/log_lls_0.dat';outpath='../../text/img/bal-recirc-auto4-success.pdf';val_from=0;val_d=5;val_to=100;rxf=-4;rxt=2;ryf=-9;ryt=2;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "bal-recirc-auto4-epoch.pdf"
awk '{if(NF==2) print $1,$2,50000; else print $0}' stats/auto4_bal_recirc/log_lle_0.dat | awk '{print $1,$2,($3>10000)?10000:$3}' > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/bal-recirc-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=10000;rxf=-4;rxt=2;ryf=-9;ryt=2;rgb_a=10;rgb_b=13;rgb_c=33" contour.p

#=================== BAL RECIRC ====================
  #=================== auto4 ====================
#awk '{if(NF<=1||(-3.5<$1 && $1<2)) print $0}' stats/auto4_bal_recirc/log_lls_0.dat > buf.dat

  echo "generec-auto4-success.pdf"
gnuplot -e "inpath='stats/auto4_generec/log_lls_0.dat';outpath='../../text/img/generec-auto4-success.pdf';val_from=0;val_d=5;val_to=100;rxf=-4;rxt=2;ryf=-7;ryt=2;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "generec-auto4-epoch.pdf"
awk '{if(NF==2) print $1,$2,50000; else print $0}' stats/auto4_generec/log_lle_0.dat | awk '{print $1,$2,($3>20000)?20000:$3}' > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/generec-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=10000;rxf=-4;rxt=2;ryf=-7;ryt=2;rgb_a=10;rgb_b=13;rgb_c=33" contour.p

#=================== auto4 ====================
ls | grep -o 'k3_139[0-9]\+_[3-9]' | sort | uniq | while read filename
do
  num=$(echo $filename | grep -o '[0-9]$')
  echo "motam $filename with num=$num"
  
  #bash zmotaj_stats.bash w $filename w 
  
  echo "tlr-k3-$num-success.pdf"
  gnuplot -e "inpath='stats/$filename/log_lls_0.dat';outpath='../../text/img/k3/tlr-$num-success.pdf';val_from=0;val_d=10;val_to=100;rxf=-4;rxt=4;ryf=-7;ryt=1;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "tlr-k3-$num-epoch.pdf"
  awk '{if(NF==2) print $1,$2,5000; else print $0}' stats/$filename/log_lle_0.dat | awk '{print $1,$2,($3>5000)?5000:$3}' > buf.dat
  gnuplot -e "inpath='buf.dat';outpath='../../text/img/k3/tlr-$num-epoch.pdf';val_from=0;val_d=300;val_to=3000;rxf=-4;rxt=4;ryf=-7;ryt=1;rgb_a=10;rgb_b=13;rgb_c=33" contour.p
done
  
#ls | grep -o 'k3_4_[0-9]_139[0-9]\+' | sort | uniq | while read filename
#do
#  echo "motam $filename"
#  bash zmotaj_stats.bash $filename $filename 
#done 

