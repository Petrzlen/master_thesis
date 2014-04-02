set   autoscale # scale axes automatically
unset log # remove any log-scaling
unset label # remove any previous labels
set xtic auto  
set ytic auto  
set terminal pdf
set key autotitle columnhead

columns = "err success h_dist h_f_b_dist m_avg_w m_sim first_second o_f_b_dist in_triangle fluctuation lambda_ih"

set title "Error to epochs"
set output 'epoch_to_err.pdf'
set xlabel "Epochs"
set ylabel "Error"
plot "epoch_to_err.dat" using 1:2 with lines, \
     "epoch_to_err.dat" using 1:3 with lines, \
     "epoch_to_err.dat" using 1:4 with lines, \
     "epoch_to_err.dat" using 1:5 with lines, \
     "epoch_to_err.dat" using 1:6 with lines
     
set title "Success to epochs"
set output 'epoch_to_success.pdf'
set xlabel "Epochs"
set ylabel "Success"
plot "epoch_to_success.dat" using 1:3 with lines
     
set title "Hidden distance to epochs"
set output 'epoch_to_h_dist.pdf'
set xlabel "Epochs"
set ylabel "Hidden distance"
plot "epoch_to_h_dist.dat" using 1:2 with lines, \
     "epoch_to_h_dist.dat" using 1:3 with lines, \
     "epoch_to_h_dist.dat" using 1:4 with lines, \
     "epoch_to_h_dist.dat" using 1:5 with lines, \
     "epoch_to_h_dist.dat" using 1:6 with lines
     
set title "Lambda to success"
set output 'success_to_lambda.pdf'
set xlabel "Success"
set ylabel "Lambda"
plot "post_success_lambda.dat" using 1:2 with lines

set title "Lambda IH to success"
set output 'success_to_lambda_ih.pdf'
set xlabel "Success"
set ylabel "Lambda IH"
plot "post_success_lambda_ih.dat" using 1:2 with lines

set title "Sigma to success"
set output 'success_to_lambda.pdf'
set xlabel "Success"
set ylabel "Lambda"
plot "post_success_sigma.dat" using 1:2 with lines
     
set title "Lambda and Sigma to success"
set output 'success_to_lambda_and_sigma.pdf'
set xlabel "Sigma"
set ylabel "Lambda"
set zlabel "Success"
splot "post_success_lambda_sigma.dat" using 1:2:3 with lines lt rgb "blue"

set dgrid3d 15,15 
set title "Lambda_IH and Lambda_HI to success"
set output 'success_to_lambdas.pdf'
set xlabel "Lambda_IH"
set logscale x
set xrange [] reverse

set ylabel "Lambda_HI"
set logscale y
set yrange [] reverse

set zlabel "Success"
set zrange [50:91] 
splot "../lambda_2d.dat" using 1:2:3 with lines
     

