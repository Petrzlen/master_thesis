#!/bin/bash


#===================== hidden activations ==========================

#bal_orig/bad/auto4_1_139846
#bad init 3116205
#non convex 3118534, 3574083
#big step 3568416, 3572476
#stagnation 3599582

#bal_orig/good/auto4_1_139846
#good init 3567401
#rect 3565457
#small 3565802, 3570398
#corner 3573063
#line-like 3566768
#travel 3566929

#tlr_can/bad/auto4_1_139846
#two static 6290231 #only bad from 500
#tlr/bad/auto4_1_139846
#tiny corner 5047426, 5047426
#two static 5050919
#weird 5072307 , 5078142
#non-convex init , 5053233
#cannot leave non-convex 5054076

#tlr_can/good/auto4_1_139846
#perfect init 5046959
#good init 5046299, 5046677
#left non-convex 5046497, 5046763
#expand 5049055, 5053958, 5050613
#one left non-convex 5050037

#=========== BAL 
echo "copying hidden activations for BAL" 
cp hr/bal_orig/bad/auto4_1_2_1398463116205.pdf  ../../text/img/hid-bal-bad-init.pdf
cp hr/bal_orig/bad/auto4_1_2_1398463574083.pdf  ../../text/img/hid-bal-bad-convex.pdf
cp hr/bal_orig/bad/auto4_1_2_1398463572476.pdf  ../../text/img/hid-bal-bad-step.pdf
cp hr/bal_orig/bad/auto4_1_2_1398463599582.pdf  ../../text/img/hid-bal-bad-stagnation.pdf

cp hr/bal_orig/good/auto4_1_2_1398463567401.pdf  ../../text/img/hid-bal-good-init.pdf
cp hr/bal_orig/good/auto4_1_2_1398463565457.pdf  ../../text/img/hid-bal-good-convex.pdf
cp hr/bal_orig/good/auto4_1_2_1398463570398.pdf  ../../text/img/hid-bal-good-step.pdf
cp hr/bal_orig/good/auto4_1_2_1398463566929.pdf  ../../text/img/hid-bal-good-stagnation.pdf

#=========== TLR 
echo "copying hidden activations for TLR" 
cp hr/tlr_can/bad/auto4_1_2_1398466290231.pdf  ../../text/img/hid-tlr-bad-static.pdf
cp hr/tlr/bad/auto4_1_2_1398465047426.pdf  ../../text/img/hid-tlr-bad-tiny.pdf
cp hr/tlr/bad/auto4_1_2_1398465054076.pdf  ../../text/img/hid-tlr-bad-init.pdf
cp hr/tlr/bad/auto4_1_2_1398465072307.pdf  ../../text/img/hid-tlr-bad-weird.pdf

cp hr/tlr/good/auto4_1_2_1398465050613.pdf  ../../text/img/hid-tlr-good-static.pdf
cp hr/tlr/good/auto4_1_2_1398465050037.pdf  ../../text/img/hid-tlr-good-tiny.pdf
cp hr/tlr/good/auto4_1_2_1398465046959.pdf  ../../text/img/hid-tlr-good-init.pdf
cp hr/tlr/good/auto4_1_2_1398465049055.pdf  ../../text/img/hid-tlr-good-weird.pdf

#===================== epoch evolution =============================
sh performance.sh 'auto4_tlrbbest' 500 0.2
gnuplot -e "outpath='../../text/img/tlr-auto4-best-perf.pdf'" performance.p

sh performance.sh 'auto4_tlr_best_bcan' 500 0.2
gnuplot -e "outpath='../../text/img/tlr-auto4-best-can.pdf'" performance.p

sh performance.sh 'k3_3_tlr_best' 500 0.2
gnuplot -e "outpath='../../text/img/tlr-k3-3-best-perf.pdf'" performance.p

sh performance.sh 'k3_3_tlr_best_can' 500 0.2
gnuplot -e "outpath='../../text/img/tlr-k3-3-best-can.pdf'" performance.p

#======================= MOMENTUM ==================
declare -a mom_arr=("0.001" "0.003" "0.01" "0.03" "0.1" "0.3")

## now loop through the above array
for mom in "${mom_arr[@]}"
do
  suf=`echo $mom | sed 's/\./-/g'`
  echo "tlr-mom-auto4-success-$suf.pdf"
  less "stats/auto4_tlr_mom/lls_0.dat" | awk "{if(\$3==$mom) print \$1,\$2,\$4;}" | bash to_log_grid.bash > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/tlr-mom-auto4-success-$suf.pdf';val_from=30;val_d=5;val_to=100;rxf=-3;rxt=8;ryf=-9;ryt=2;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

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
gnuplot -e "inpath='stats/auto4_bal_orig/log_lls_0.dat';outpath='../../text/img/tlr-auto4-success.pdf';val_from=40;val_d=5;val_to=100;rxf=-4;rxt=9;ryf=-9;ryt=1;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "tlr-auto4-epoch.pdf"
  less stats/auto4_bal_orig/log_lle_0.dat | bash post_epochs.bash 20000 > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/tlr-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=5000;rxf=-4;rxt=9;ryf=-9;ryt=1;rgb_a=10;rgb_b=13;rgb_c=33" contour.p

  #=================== digits ===================

  echo "tlr-digits-error.pdf"
gnuplot -e "inpath='stats/digits_tlr/log_llp_0.dat';outpath='../../text/img/tlr-digits-psf.pdf';val_from=0.5;val_d=0.05;val_to=1.0;rxf=-2;rxt=6;ryf=-9;ryt=-2;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "tlr-digits-epoch.pdf"
gnuplot -e "inpath='stats/digits_tlr/log_lle_0.dat';outpath='../../text/img/tlr-digits-epoch.pdf';val_from=7;val_d=1;val_to=15;rxf=-2;rxt=6;ryf=-9;ryt=-2;rgb_a=10;rgb_b=13;rgb_c=33" contour.p

#=================== BAL RECIRC ====================
  #=================== auto4 ====================
#awk '{if(NF<=1||(-3.5<$1 && $1<2)) print $0}' stats/auto4_bal_recirc/log_lls_0.dat > buf.dat

  echo "bal-recirc-auto4-success.pdf"
gnuplot -e "inpath='stats/auto4_bal_recirc/log_lls_0.dat';outpath='../../text/img/bal-recirc-auto4-success.pdf';val_from=0;val_d=5;val_to=100;rxf=-4;rxt=2;ryf=-9;ryt=2;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "bal-recirc-auto4-epoch.pdf"
  less stats/auto4_bal_recirc/log_lle_0.dat | bash post_epochs.bash 20000 > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/bal-recirc-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=5000;rxf=-4;rxt=2;ryf=-9;ryt=2;rgb_a=10;rgb_b=13;rgb_c=33" contour.p

#=================== BAL RECIRC ====================
  #=================== auto4 ====================
#awk '{if(NF<=1||(-3.5<$1 && $1<2)) print $0}' stats/auto4_bal_recirc/log_lls_0.dat > buf.dat

  echo "generec-auto4-success.pdf"
gnuplot -e "inpath='stats/auto4_generec/log_lls_0.dat';outpath='../../text/img/generec-auto4-success.pdf';val_from=0;val_d=5;val_to=100;rxf=-4;rxt=2;ryf=-7;ryt=2;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "generec-auto4-epoch.pdf"
less stats/auto4_generec/log_lle_0.dat | bash post_epochs.bash 20000 > buf.dat
gnuplot -e "inpath='buf.dat';outpath='../../text/img/generec-auto4-epoch.pdf';val_from=0;val_d=1000;val_to=8000;rxf=-4;rxt=2;ryf=-7;ryt=2;rgb_a=10;rgb_b=13;rgb_c=33" contour.p

#==================== HISTORY =====================
  #=================== auto4 ====================
ls 'stats' | grep -o 'stats/k3_139[0-9]\+_[3-9]' | sort | uniq | while read filename
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
  

#=================== auto4 ====================
ls 'stats' | grep -o 'k3_4_[3-9]_1398[0-9]\+' | sort | uniq | while read filename
do
  num=$(echo $filename | grep -o 'k3_4_[0-9]' | grep -o '[0-9]$')
  echo "motam $filename with num=$num"
  
  #bash zmotaj_stats.bash w $filename w 
  
  echo "  generec-k3-$num-success.pdf"
  gnuplot -e "inpath='stats/$filename/log_lls_0.dat';outpath='../../text/img/k3/generec-$num-success.pdf';val_from=0;val_d=10;val_to=100;rxf=-4;rxt=4;ryf=-7;ryt=1;rgb_a=33;rgb_b=13;rgb_c=10" contour.p

  echo "  generec-k3-$num-epoch.pdf"
  less stats/$filename/log_lle_0.dat | bash post_epochs.bash 5000 > buf.dat
  gnuplot -e "inpath='buf.dat';outpath='../../text/img/k3/generec-$num-epoch.pdf';val_from=0;val_d=300;val_to=2500;rxf=-4;rxt=4;ryf=-7;ryt=1;rgb_a=10;rgb_b=13;rgb_c=33" contour.p
done
