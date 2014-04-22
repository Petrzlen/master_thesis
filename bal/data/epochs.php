<?php
// Report simple running errors
error_reporting(E_ERROR | E_WARNING | E_PARSE);

//TODO gnuplot stddev 

//$files = array('stats/two_lambdas_long/auto4_1396525966589_2_post.csv', 'stats/two_lambdas_long/auto4_1396710717778_2_post.csv', 'stats/two_lambdas_long/auto4_1396865043724_2_post.csv'); 

parse_str(implode('&', array_slice($argv, 1)), $_GET);
$files = $_GET['files']; 

$iep = isset($_GET['epoch_id']) ? $_GET['epoch_id'] : 0; 
$ier = isset($_GET['err_id']) ? $_GET['err_id'] : 1; 
$il1 = isset($_GET['l1_id']) ? $_GET['l1_id'] : 3; 
$il2 = isset($_GET['l2_id']) ? $_GET['l2_id'] : 13; 

foreach($files as $f => $file){
  $data = file_get_contents($file); 
  $lines = explode("\n", $data); 
  $first = true; 
  
  /*
  foreach($lines as $l => $line){
    if($first) {
      $first = false; 
      continue; 
    }
    $arr = explode(" ", $line);  
    $max_epoch = max($arr[$iep], $max_epoch);
  } */
  
  //echo 'max_epoch='.$max_epoch; 
  
  foreach($lines as $l => $line){
    if($first) {
      $first = false; 
      continue; 
    }

    $arr = explode(" ", $line);  
    if(!isset($arr[$ier])) continue; 
    //print_r($arr); 
    
    if($arr[$ier] == 0.0){ //&& $arr[$iep] < $max_epoch){
      $f1 = (float) $arr[$il1]; 
      $f2 = (float) $arr[$il2]; 
      $e = (float) $arr[$iep]; 
      $S[''.$f1][''.$f2] += $e;
      $C[''.$f1][''.$f2] ++;
      
      //echo $f1.' '.$arr[$il1]."\n";
    } 
  }
}

//print_r($S); 
//print_r($C); 

echo "lambda\tlambda_ih\tepoch\n"; 
foreach($S as $l1 => $line){
  foreach($line as $l2 => $s){
    $c = $C[$l1][$l2];
    if($c > 0){ //how many samples need to be 
      echo "$l1\t$l2\t".($s / $c)."\n";
    }
  }
}


/*
epoch 
err 
sigma 
lambda 
momentum 
h_dist 
h_f_b_dist 
m_avg_w 
m_sim 
first_second 
o_f_b_dist 
in_triangle 
fluctuation 
lambda_ih
*/
?>
