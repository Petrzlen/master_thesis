#!/bin/bash

measure="measure.csv" 
pre="post.csv" 
post="post.csv" 

if [ -z "$3" ]
then
  echo "Creating $2 for $1" 
  mkdir stats/$2 
  cd  stats/$2

  cp "../../"$1"_measure.csv" "measure.csv"
  cp "../../"$1"_pre.csv"     "pre.csv" 
  cp "../../"$1"_post.csv"    "post.csv" 

  #distribution of epochs needed to settle into no error state 
  echo "Calculating convergence_epochs" 
  less $post | awk '{print $1}' | tail -n +2 > convergence_epochs.dat 

  echo "DROP TABLE data;\nCREATE TABLE data (\n  run_id TEXT," > create_table.sql
  head -1 $measure | sed 's/\t/,\n/g' | tail -n +2 | sed 's/\(^[^,]*\)/  \1 DOUBLE/g' >> create_table.sql 
  echo ");" >> create_table.sql 

  less $measure | grep '\.' | sed 's/\t/","/g' | sed 's/\(.*\)/INSERT INTO data VALUES ("\1");/g' > insert_table.sql

  echo "ALTER TABLE data ADD success INT; UPDATE data SET success = (CASE WHEN err = 0.0 THEN 1 ELSE 0 END); CREATE INDEX index_err ON data (err); CREATE INDEX index_success ON data (success); CREATE INDEX index_epoch ON data (epoch);" > update_table.sql

  sqlite3 measure.sqlite < create_table.sql
  echo "Created table" 
  sqlite3 measure.sqlite < insert_table.sql
  echo "Inserted data" 
  sqlite3 measure.sqlite < update_table.sql
  echo "Updated data" 
else
  echo "Using existing $2 for $1"
  cd  stats/$2
fi 

# =========== RUN SQLs on the created database ================== " 

# sum of error after each epoch 
#tail -n +2 auto4_$1\_2_measure.dat | awk '{a[$2] += $3}END{for(epoch in a) print epoch,a[epoch]}' | sort -n | tail -n +100 > $2/epochs_sum.dat
#declare -a arr=("err" "success" "h_dist" "h_f_b_dist" "m_avg_w" "m_sim" "first_second" "o_f_b_dist" "in_triangle" "fluctuation")

## now loop through the above array
for i in "err" "success" "h_dist" "h_f_b_dist" "m_avg_w" "m_sim" "first_second" "o_f_b_dist" "in_triangle" "fluctuation" "lambda_ih" 
do
  #TODO optimize 
  echo "epoch to $i" 
  sqlite3 measure.sqlite <<< "SELECT cast(D1.epoch as int) AS 'epoch', MIN(D1.$i) AS 'all_min_$i', AVG(D1.$i) AS 'all_avg_$i', MAX(D1.$i) AS 'all_max_$i', (SELECT AVG(D2.$i) FROM data D2 WHERE D1.epoch=D2.epoch AND D2.success=0) AS 'bad_avg_$i', (SELECT AVG(D2.$i) FROM data D2 WHERE D1.epoch=D2.epoch AND D2.success=1) AS 'good_avg_$i' FROM data D1 GROUP BY D1.epoch;" | sed 's/|/\t/g' > epoch_to_$i.dat 
# ========== GROUP BY success 
#  sqlite3 measure.sqlite <<< "SELECT cast(epoch as int) AS 'E', AVG(SELECT $1 FROM data WHERE FROM data GROUP BY epoch;" | sed 's/|/\t/g' > good_to_bad_$1.dat 
done

#========== POST MEASURES (success) ============
echo "post_success_lambda" 
sqlite3 measure.sqlite <<< "SELECT lambda, AVG(success) AS 'success' FROM data WHERE epoch = (SELECT MAX(epoch) FROM data) GROUP BY lambda;" | sed 's/|/\t/g' > post_success_lambda.dat 

echo "post_success_lambda_ih" 
sqlite3 measure.sqlite <<< "SELECT lambda_ih, AVG(success) AS 'success' FROM data WHERE epoch = (SELECT MAX(epoch) FROM data) GROUP BY lambda_ih;" | sed 's/|/\t/g' > post_success_lambda_ih.dat 

echo "post_success_sigma" 
sqlite3 measure.sqlite <<< "SELECT sigma, AVG(success) AS 'success' FROM data WHERE epoch = (SELECT MAX(epoch) FROM data) GROUP BY sigma;" | sed 's/|/\t/g' > post_success_sigma.dat 

#TODO several group by 
echo "post_success_lambda_sigma" 
sqlite3 measure.sqlite <<< "SELECT lambda, sigma, AVG(success) AS 'success' FROM data WHERE epoch = (SELECT MAX(epoch) FROM data) GROUP BY lambda,sigma;" | sed 's/|/\t/g' > post_success_lambda_sigma.dat 

# measure.column distributions (e.g. ten uniformly distributed interval buckets) 
  # at end of the run 
  
# hidden dist depending on the error 
# SELECT err, AVG(h_dist) FROM data WHERE epoch = (SELECT MAX(epoch) FROM data) GROUP BY err;

# ======== MUTLI DIMENSIONAL DATA ======== (TODO) 
# GNUPLOT: splot "./bal/data/hdist_stats_0.csv" using 1:2:3 with lines lt rgb "blue"
# sample file: 
##sigma  lambda  success
#1       0.001   0.4
#1       0.003   8.75
#1       0.01    30.8
#1       0.03    47.03389830508475
#1       0.1     54.112554112554115
#1       0.3     56.97211155378486
#1       1       57.3394495412844
#1.3     0.001   0.8264462809917356
#1.3     0.003   14.354066985645932
#1.3     0.01    32.5

# ========== PRE MEASURE vs POST MEASURE ===========

# do some correlations and covariances of measures 

#TODO group by (epoch, success) (i.e. good vs. bad) and (epoch, err) -> make columns for each value of the second

# ============ PLOT the data ================ 
echo "plotting data" 
gnuplot ../../panko_plot.p 

#TODO compare several models 

cd ../../

