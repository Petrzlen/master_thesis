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
head -1 $measure | sed 's/\t/\n/g' | tail -n +2 > cols.txt
echo "success" >> cols.txt
#for i in "err" "success" "h_dist" "h_f_b_dist" "m_avg_w" "m_sim" "first_second" "o_f_b_dist" "in_triangle" "fluctuation" "lambda_ih" 
cat cols.txt | while read i
do
  #TODO optimize (stddev sucks)
  echo "epoch to '$i'" 
  sqlite3 measure.sqlite <<< "SELECT cast(D1.epoch as int) AS 'epoch', MIN(D1.$i) AS 'all_min_$i', MAX(D1.$i) AS 'all_max_$i', AVG(D1.$i) AS 'all_avg_$i', AVG((D1.$i - (SELECT AVG(D2.$i) FROM data D2 WHERE D1.epoch=D2.epoch)) * (D1.$i - (SELECT AVG(D2.$i) FROM data D2 WHERE D1.epoch=D2.epoch))) AS 'all_stdevp_$i', (SELECT AVG(D2.$i) FROM data D2 WHERE D1.epoch=D2.epoch AND D2.success=0) AS 'bad_avg_$i', (SELECT AVG(D2.$i) FROM data D2 WHERE D1.epoch=D2.epoch AND D2.success=1) AS 'good_avg_$i', (SELECT AVG(D2.$i) FROM data D2 WHERE D1.epoch=D2.epoch) AS 'a' FROM data D1 GROUP BY D1.epoch;" | sed 's/|/\t/g' > epoch_$i.dat 
done

#========== POST MEASURES (success) ============
cat cols.txt | while read i
do
  echo "'$i' to success" 
  sqlite3 measure.sqlite <<< "SELECT $i, AVG(success) AS 'success' FROM data WHERE epoch = (SELECT MAX(epoch) FROM data) GROUP BY $i;" | sed 's/|/\t/g' > $i"_success.dat"
done

fi=0
ls . | grep 'post.csv' | while read f
do
  echo "$f : lambdah_lambdav_success.dat" 
  sqlite3 measure.sqlite <<< "SELECT lambda, lambda_ih, AVG(success) AS 'success' FROM data WHERE epoch = (SELECT MAX(epoch) FROM data) GROUP BY lambda,lambda_ih;" | sed 's/|/\t/g' > lambdah_lambdav_success_$fi.dat 

  echo "$f : lambdah_lambdav_epoch.dat" 
  php ../../epochs.php files[]=$f > lambdah_lambdav_epoch_$fi.dat
  fi=$fi+1
done 

# ============ PLOT the data ================ 
echo "plotting data" 
gnuplot ../../panko_plot.p 

cd ../../

