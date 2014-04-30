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

  rm measure.sqlite 
  #echo "DROP TABLE data;" > create_table.sql
  echo "CREATE TABLE data (" > create_table.sql
  echo "  run_id TEXT," >> create_table.sql
  head -1 $measure | sed 's/\t/,\n/g' | tail -n +2 | sed 's/\(^[^,]*\)/  \1 DOUBLE/g' >> create_table.sql 
  echo ");" >> create_table.sql 

  rm insert_table.sql 
  less $measure | grep '\.' | sed 's/\t/","/g' | sed 's/\(.*\)/INSERT INTO data VALUES ("\1");/g' >> insert_table.sql
  #fix no measure at end 
  less $post | grep '\.' | sed 's/\t/","/g' | sed 's/\(.*\)/INSERT INTO data VALUES ("NA","\1");/g' >> insert_table.sql

  echo "ALTER TABLE data ADD success INT; UPDATE data SET success = (CASE WHEN err = 0.0 THEN 1 ELSE 0 END); CREATE INDEX index_err ON data (err); CREATE INDEX index_success ON data (success); CREATE INDEX index_epoch ON data (epoch); ALTER TABLE data ADD buf DOUBLE;" > update_table.sql

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

## now loop through the above array
head -1 $measure | sed 's/\t/\n/g' | tail -n +2 > cols.txt
echo "success" >> cols.txt
cat cols.txt | while read i
do
  echo "epoch to '$i'" 
  sqlite3 measure.sqlite "SELECT load_extension('./../../libsqlitefunctions.so'); SELECT cast(D1.epoch as int) AS 'epoch', COUNT(*) AS 'count', MIN(D1.$i) AS 'all_min_$i', MAX(D1.$i) AS 'all_max_$i', AVG(D1.$i) AS 'all_avg_$i', STDEV(D1.$i) AS 'all_stdev_$i', (SELECT AVG(D2.$i) FROM data D2 WHERE D1.epoch=D2.epoch AND D2.success=0) AS 'bad_avg_$i', (SELECT AVG(D2.$i) FROM data D2 WHERE D1.epoch=D2.epoch AND D2.success=1) AS 'good_avg_$i' FROM data D1 GROUP BY D1.epoch;" | sed 's/|/\t/g' | tail -n +3 > epoch_$i.dat 
done

#========== POST MEASURES (success) ============
cat cols.txt | while read i
do
  echo "'$i' to success" 
  sqlite3 measure.sqlite <<< "SELECT $i, AVG(success) AS 'success' FROM data WHERE epoch <> 0 GROUP BY $i;" | sed 's/|/\t/g' > $i"_success.dat"
done

fi=0
ls . | grep 'post.csv' | while read f
do
  echo "$f : lambdah_lambdav_success.dat" 
  
  #OLDEST TIMES ^^ 
  if sqlite3 measure.sqlite <<< ".schema" | grep -q 'lambda_ih'; then
     
    sqlite3 measure.sqlite <<< "SELECT lambda, lambda_ih, 100*AVG(success) AS 'success' FROM data WHERE epoch <> 0 GROUP BY lambda,lambda_ih;" | sed 's/|/\t/g' > lls_$fi.dat 

    echo "$f : lambdah_lambdav_success.dat" 
    
    sqlite3 measure.sqlite <<< "SELECT D1.lambda, D1.lambda_ih, (SELECT AVG(D2.epoch) FROM data D2 WHERE D2.epoch <> 0 AND D2.success = 1 AND D1.lambda=D2.lambda AND D1.lambda_ih=D2.lambda_ih) as 'epoch' FROM data D1 GROUP BY lambda,lambda_ih;" | sed 's/|/\t/g' > lle_$fi.dat 
  #NEW TIMES 
  else
    lv="lam"
    lh="lam_v" #not so old times 
    if sqlite3 measure.sqlite <<< ".schema" | grep -q 'lam_h'; then
      lv="lam_v" 
      lh="lam_h" 
    fi 
    
    sqlite3 measure.sqlite <<< "SELECT $lv as 'lambda_v', $lh as 'lambda_h', 100*AVG(success) AS 'success' FROM data WHERE epoch <> 0 GROUP BY $lv,$lh;" | sed 's/|/\t/g' > lls_$fi.dat 
    #sqlite3 measure.sqlite <<< "SELECT lam, lam_v, mom, 100*AVG(success) AS 'success' FROM data WHERE epoch <> 0 GROUP BY lam,lam_v,mom;" | sed 's/|/\t/g' > lls_$fi.dat 

    #SELECT lambda, lambda_ih, 100*AVG(success) AS 'suc', AVG(epoch) AS 'epc' FROM data WHERE epoch <> 0 GROUP BY lambda,lambda_ih ORDER BY suc ASC, epc DESC;

    echo "$f : lambdah_lambdav_ps_f.dat" 

    sqlite3 measure.sqlite <<< "SELECT $lv as 'lambda_v', $lh as 'lambda_h', MIN(ps_f) AS 'ps_f' FROM data GROUP BY $lv,$lh;" | sed 's/|/\t/g' > llp_$fi.dat 
    
    echo "$f : lambdah_lambdav_epoch.dat" 

    #wasted work :/ 
    #php ../../epochs.php files[]=$f > lambdah_lambdav_epoch_$fi.dat
    #php ../../epochs.php files[]=$f l1_id=2 l2_id=3 > lle_$fi.dat
    
    sqlite3 measure.sqlite <<< "SELECT D1.$lv as 'lambda_v', D1.$lh as 'lambda_h', (SELECT AVG(D2.epoch) FROM data D2 WHERE D2.epoch <> 0 AND D2.success = 1 AND D1.$lv=D2.$lv AND D1.$lh=D2.$lh) as 'epoch' FROM data D1 GROUP BY $lv,$lh;" | sed 's/|/\t/g' > lle_$fi.dat 
    
    #Momentu
    #sqlite3 measure.sqlite <<< "SELECT D1.lam, D1.lam_v, D1.mom, (SELECT AVG(D2.epoch) FROM data D2 WHERE D2.epoch <> 0 AND D2.success = 1 AND D1.lam=D2.lam AND D1.lam_v=D2.lam_v AND D1.mom=D2.mom) as 'epoch' FROM data D1 GROUP BY lam,lam_v,mom;" | sed 's/|/\t/g' > lle_$fi.dat
    
#    echo "SELECT D1.$lv as 'lambda_v', D1.$lh as 'lambda_h', MAX(D1.epoch) AS 'epoch' FROM data D1 GROUP BY $lv,$lh;"
    
    #digits 
    #sqlite3 measure.sqlite <<< "SELECT D1.$lv as 'lambda_v', D1.$lh as 'lambda_h', MAX(D1.epoch) AS 'epoch' FROM data D1 GROUP BY $lv,$lh;" | sed 's/|/\t/g' > lle_$fi.dat  
  fi
  
  awk 'BEGIN{last=-42;}{if(NR==1 || $1==0 || $2==0) next; if(last!=-42 && last!=$1) {printf "\n";} last=$1; print(log($1)/log(10), log($2)/log(10), $3);}' lls_$fi.dat > log_lls_$fi.dat 
  
  awk 'BEGIN{last=-42;}{if(NR==1 || $1==0 || $2==0) next; if(last!=-42 && last!=$1) {printf "\n";} last=$1; print(log($1)/log(10), log($2)/log(10), $3);}' lle_$fi.dat > log_lle_$fi.dat 
  
  awk 'BEGIN{last=-42;}{if(NR==1 || $1==0 || $2==0) next; if(last!=-42 && last!=$1) {printf "\n";} last=$1; print(log($1)/log(10), log($2)/log(10), $3);}' llp_$fi.dat > log_llp_$fi.dat 
  
  fi=$fi+1
done 

cd ../../

