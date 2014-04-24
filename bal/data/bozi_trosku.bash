#!/bin/bash

#=================== TLR ====================
  #=================== auto4 ====================
  echo "tlr-auto4-success.pdf"
gnuplot -e "inpath='stats/auto4_bal_orig/log_lls_0.dat';outpath='../../text/img/tlr-auto4-success.pdf';val_from=0;val_d=5;val_to=100;rxf=-4;rxt=9;ryf=-9;ryt=1;" contour.p

  echo "tlr-auto4-epoch.pdf"
awk '{if(NF==2) print $1,$2,50000; else print $0}' stats/auto4_bal_orig/log_lle_0.dat | awk '{print $1,$2,($3>10000)?10000:$3}' > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/tlr-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=10000;rxf=-4;rxt=9;ryf=-9;ryt=1;" contour.p

#=================== BAL RECIRC ====================
  #=================== auto4 ====================
#awk '{if(NF<=1||(-3.5<$1 && $1<2)) print $0}' stats/auto4_bal_recirc/log_lls_0.dat > buf.dat

  echo "bal-recirc-auto4-success.pdf"
gnuplot -e "inpath='stats/auto4_bal_recirc/log_lls_0.dat';outpath='../../text/img/bal-recirc-auto4-success.pdf';val_from=0;val_d=5;val_to=100;rxf=-4;rxt=2;ryf=-9;ryt=2;" contour.p

  echo "bal-recirc-auto4-epoch.pdf"
awk '{if(NF==2) print $1,$2,50000; else print $0}' stats/auto4_bal_recirc/log_lle_0.dat | awk '{print $1,$2,($3>10000)?10000:$3}' > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/bal-recirc-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=10000;rxf=-4;rxt=2;ryf=-9;ryt=2;" contour.p

#=================== BAL RECIRC ====================
  #=================== auto4 ====================
#awk '{if(NF<=1||(-3.5<$1 && $1<2)) print $0}' stats/auto4_bal_recirc/log_lls_0.dat > buf.dat

  echo "generec-auto4-success.pdf"
gnuplot -e "inpath='stats/auto4_generec/log_lls_0.dat';outpath='../../text/img/generec-auto4-success.pdf';val_from=0;val_d=5;val_to=100;rxf=-4;rxt=2;ryf=-7;ryt=2;" contour.p

  echo "generec-auto4-epoch.pdf"
awk '{if(NF==2) print $1,$2,50000; else print $0}' stats/auto4_generec/log_lle_0.dat | awk '{print $1,$2,($3>20000)?20000:$3}' > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/generec-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=10000;rxf=-4;rxt=2;ryf=-7;ryt=2;" contour.p

ls | grep -o 'k3_139[0-9]\+_[0-9]\+' | sort | uniq | while read filename
do
  echo "motam $filename"
  bash zmotaj_stats.bash $filename $filename 
done 

