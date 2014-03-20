#! /bin/bash

#TODO SQLITE 

mkdir stats/$2 
cd  stats/$2

measure = ../../auto4_$1\_2_measure.dat  
pre = ../../auto4_$1\_2_pre.dat  
post = ../../auto4_$1\_2_post.dat  

#TODO learning rate & sigma dependent (group by) 

#distribution of epochs needed to settle into no error state 
echo "Calculating convergence_epochs" 
less $post | awk '{print $1}' | tail -n +2 > $2/convergence_epochs.dat 

echo "DROP TABLE data;\nCREATE TABLE data (\n  run_id TEXT," > create_table.sql
head -1 $measure | sed 's/\t/,\n/g' | tail -n +2 | sed 's/\(^[^,]*\)/  \1 DOUBLE/g' >> create_table.sql 
echo ");" >> create_table.sql 

less $measure | grep '\.' | sed 's/\t/","/g' | sed 's/\(.*\)/INSERT INTO data VALUES ("\1");/g' > insert_table.sql

echo "ALTER TABLE data ADD success INT; UPDATE data SET success = (CASE WHEN err = 0.0 THEN 0 ELSE 1 END);" > update_table.sql

#sqlite3 measure.sqlite < create_table.sql
echo "Created table" 
#sqlite3 measure.sqlite < insert_table.sql
echo "Inserted data" 
sqlite3 measure.sqlite < update_table.sql
echo "Updated data" 

# =========== RUN SQLs on the created database ================== " 

# sum of error after each epoch 
#tail -n +2 auto4_$1\_2_measure.dat | awk '{a[$2] += $3}END{for(epoch in a) print epoch,a[epoch]}' | sort -n | tail -n +100 > $2/epochs_sum.dat

declare -a arr=("err", "success",	"h_dist",	"h_f_b_dist","m_avg_w","m_sim", "first_second","o_f_b_dist","in_triangle","fluctuation")
for i in "${arr[@]}"
do
  sqlite3 measure.sqlite <<< "SELECT cast(epoch as int) AS 'epoch', MIN($1) AS 'min_$1', AVG($1) AS 'avg_$1', MAX($1) AS 'max_$1' FROM data GROUP BY epoch;" | sed 's/|/\t/g' > epoch_to_$1.dat 
# ========== GROUP BY success 
#  sqlite3 measure.sqlite <<< "SELECT cast(epoch as int) AS 'E', AVG(SELECT $1 FROM data WHERE FROM data GROUP BY epoch;" | sed 's/|/\t/g' > good_to_bad_$1.dat 
done

#========== POST MEASURES (success) ============
sqlite3 measure.sqlite <<< "SELECT lambda, AVG(success) AS 'success' FROM data WHERE epoch = (SELECT MAX(epoch) FROM data) GROUP BY lambda;" | sed 's/|/\t/g' > post_success_lambda.dat 
sqlite3 measure.sqlite <<< "SELECT sigma, AVG(success) AS 'success' FROM data WHERE epoch = (SELECT MAX(epoch) FROM data) GROUP BY sigma;" | sed 's/|/\t/g' > post_success_sigma.dat 
sqlite3 measure.sqlite <<< "SELECT lambda, sigma, AVG(success) AS 'success' FROM data WHERE epoch = (SELECT MAX(epoch) FROM data) GROUP BY lambda,sigma;" | sed 's/|/\t/g' > post_success_lambda_sigma.dat 

# measure.column distributions (e.g. ten uniformly distributed interval buckets) 
  # at end of the run 
  
# hidden dist depending on the error 
SELECT err, AVG(h_dist) FROM data WHERE epoch = (SELECT MAX(epoch) FROM data) GROUP BY err;

# ========== PRE MEASURE vs POST MEASURE ===========

# do some correlations and covariances of measures 

#TODO group by (epoch, success) (i.e. good vs. bad) and (epoch, err) -> make columns for each value of the second

# ============ PLOT the data ================ 

#TODO compare several models 

cd ../../

