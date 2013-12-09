//TODO datasety z clanku  
//TODO graf skrytych reprezentacii v priebehu (mozno 3d ciary)
//TODO dropout? 
//TODO matematicky pohlad - o'really clanok aproximacia gradientu chyby
//TODO rekonstrukcia (zmena zopar bitov, ci tam-speat da orig) 
//TODO vyssi rozmer tasku (8-3-8), (16-4-16)
//TODO kvazi momentum -> odtlacanie hidden reprezentacii, -\delta w(t-1) 
//TODO pocet epoch potrebnych na konvergenciu
//TODO nie autoassoc ale permutovat vystupy (napr. 1000 na 0100)
//TODO reprezentacia SUC/ERR 
//TODO m_sim, h_f_b_dist -> jeden nasobkom druheho bias
//TODO spektralna nalyza
//TODO bipolarna [-1, 1] vstupy
//h_size = 3 => [9,10,1] for errors [0.0, 1.0, 2.0] 

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Scanner;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class BAL {
	private static PrintWriter log = null; 
	
	public static  boolean MEASURE_IS = true; 
	public static boolean MEASURE_SAVE_AFTER_EACH_RUN = false; 
	public static  int MEASURE_RECORD_EACH = 1000;

	public static  String INPUT_FILEPATH = "auto4.in"; 
	public static  String OUTPUT_FILEPATH = "auto4.in"; 
	public static  int INIT_HIDDEN_LAYER_SIZE = 2 ; 

	public static  double CONVERGENCE_WEIGHT_EPSILON = 0.0; 
	//there was no change in given outputs for last CONVERGENCE_NO_CHANGE_FOR
	public static  int CONVERGENCE_NO_CHANGE_FOR = 10; 
	public static  double CONVERGENCE_NO_CHANGE_EPSILON = 0.001;
	public static  int INIT_MAX_EPOCHS = 30000;

	public static  int INIT_RUNS = 100; 
	public static  int INIT_CANDIDATES_COUNT = 100;

	public static  boolean PRINT_NETWORK_IS = false; 

	public static  double TRY_NORMAL_DISTRIBUTION_SIGMA[] = {2.3}; 
	//public static  double TRY_NORMAL_DISTRIBUTION_SIGMA[] = {1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2}; 
	public static  double TRY_LAMBDA[] = {0.7}; 
	//public static  double TRY_LAMBDA[] = {0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2}; 
	//public static  double TRY_LAMBDA[] = {0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5}; 
	//public static  double TRY_NOISE_SPAN[] = {0.0, 0.003, 0.01, 0.03, 0.1, 0.3}; 
	//public static  double TRY_MULTIPLY_WEIGHTS[] = {1.0, 1.00001, 1.00003, 1.0001, 1.0003, 1.001}; 
	public static  double TRY_MOMENTUM[] = {-1, -0.3, -0.1, -0.03, -0.01, -0.003, -0.001, 0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}; 

	public static  double INIT_NORMAL_DISTRIBUTION_MU = 0;
	public static  double NORMAL_DISTRIBUTION_SPAN = 15; 

	public static  String[] MEASURE_HEADINGS = {"epoch","err","h_dist","h_f_b_dist","m_avg_w","m_sim", "first_second", "sigma", "lambda", "o_f_b_dist", "momentum"}; 

	//TODO activation on hidden networks 
	//epoch
	public static  int MEASURE_EPOCH = 0; 

	//error function (RMSE) 
	public static  int MEASURE_ERROR = 1;

	//avg of dist(h_i - h_j) i \neq j where h_i is a hidden activation for input i
	//intuitively: internal representation difference 
	public static  int MEASURE_HIDDEN_DIST = 2;

	//avg distance between forward and backward activations on hidden layer
	public static  int MEASURE_HIDDEN_FOR_BACK_DIST = 3;

	//avg weight of matrixes 
	public static  int MEASURE_MATRIX_AVG_W = 4;

	//sum of |a_{ij} - b_{ij}| per pairs (HO, HI) and (OH, IH) 
	public static  int MEASURE_MATRIX_SIMILARITY = 5;

	//ratio of (a_1, a_2) where a_i is the i-th biggest output 
	public static  int MEASURE_FIRST_SECOND_RATIO = 6;

	public static  int MEASURE_SIGMA = 7; 
	public static  int MEASURE_LAMBDA = 8; 
	//public static  int MEASURE_NOISE_SPAN = 9; 
	//public static  int MEASURE_MULTIPLY_WEIGHTS = 9; 

	//avg distance between forward and backward activations on their output layers (forward layer 2, backward layer 0) 
	public static  int MEASURE_OUTPUT_FOR_BACK_DIST = 9;

	public static  int MEASURE_MOMENTUM = 10;

	public static  int MEASURE_COUNT = 11; 

	public static  int[] MEASURE_GROUP_BY_COLS = {MEASURE_ERROR, MEASURE_SIGMA, MEASURE_LAMBDA, MEASURE_MOMENTUM};
	public static  int MEASURE_GROUP_BY = MEASURE_ERROR;  

	public static Random random = new Random(); 
	public static double INIT_NORMAL_DISTRIBUTION_SIGMA = 1.25; 
	public static double INIT_LAMBDA = 0.03; 
	public static double INIT_MOMENTUM = 0.1; 
	//public static double INIT_NOISE_SPAN = 0.00; 
	//public static double INIT_MULTIPLY_WEIGHTS = 1.001; 

	private ArrayList<Double>[] measures = null; 

	public static ArrayList<double[]> pre_measure = new ArrayList<double[]>();
	public static ArrayList<double[]> post_measure = new ArrayList<double[]>();

	private RealMatrix IH; 
	private RealMatrix HO;  
	private RealMatrix OH; 
	private RealMatrix HI;

	//Momentum matrices 
	private double[][] MOM_IH; 
	private double[][] MOM_HO;  
	private double[][] MOM_OH; 
	private double[][] MOM_HI;
	
	private static String RUN_ID = null;  

	public static void run(){
		BAL.INIT_NORMAL_DISTRIBUTION_SIGMA = BAL.TRY_NORMAL_DISTRIBUTION_SIGMA[random.nextInt(BAL.TRY_NORMAL_DISTRIBUTION_SIGMA.length)]; 
		BAL.INIT_LAMBDA = BAL.TRY_LAMBDA[random.nextInt(BAL.TRY_LAMBDA.length)];
		BAL.INIT_MOMENTUM = BAL.TRY_MOMENTUM[random.nextInt(BAL.TRY_MOMENTUM.length)];
		//BAL.INIT_NOISE_SPAN = BAL.TRY_NOISE_SPAN[random.nextInt(BAL.TRY_NOISE_SPAN.length)];
		//BAL.INIT_MULTIPLY_WEIGHTS = BAL.TRY_MULTIPLY_WEIGHTS[random.nextInt(BAL.TRY_MULTIPLY_WEIGHTS.length)];

		int h_size = BAL.INIT_HIDDEN_LAYER_SIZE; 
		double lambda = BAL.INIT_LAMBDA; 
		int max_epoch = BAL.INIT_MAX_EPOCHS; 

		RealMatrix inputs = BAL.loadFromFile(BAL.INPUT_FILEPATH);
		RealMatrix outputs = BAL.loadFromFile(BAL.OUTPUT_FILEPATH);

		//select the network with the biggest hidden distance 
		double mav=0.0; 
		BAL network = null; 
		for(int i=0; i<BAL.INIT_CANDIDATES_COUNT; i++){
			BAL N = new BAL(inputs.getColumnDimension(), h_size, outputs.getColumnDimension()); 
			double[] measure = N.measure(0, inputs, outputs); 
			double hd =  measure[BAL.MEASURE_HIDDEN_DIST];
			if(hd > mav){
				mav = hd;
				network = N; 
			}
		}

		//log.println(network.printNetwork()); 

		ArrayList<Integer> order = new ArrayList<Integer>(inputs.getRowDimension());
		for(int i=0; i<inputs.getRowDimension() ; i++){
			order.add(i); 
		}

		if(MEASURE_IS) { 
			pre_measure.add(network.measure(0, inputs, outputs));
		}

		//HISTORY, INPUT_ID, VECTOR_ENTRY
		RealVector[][] given = new RealVector[CONVERGENCE_NO_CHANGE_FOR][outputs.getRowDimension()];
		int epochs=0;

		for(epochs=0; epochs<max_epoch ; epochs++){
			if(MEASURE_IS && (MEASURE_SAVE_AFTER_EACH_RUN && epochs % BAL.MEASURE_RECORD_EACH == 0)){
				//log.println(network.evaluate(inputs, outputs));
				network.measure(epochs, inputs, outputs);
			}

			//TODO Consider as a MEASURE (avg_weight_change) 
			double avg_weight_change = 0.0; 
			java.util.Collections.shuffle(order);

			for(int order_i = 0; order_i < order.size() ; order_i++){
				RealVector in = inputs.getRowVector(order_i);
				RealVector out = outputs.getRowVector(order_i);

				avg_weight_change += network.learn(in, out, lambda);
				given[epochs % CONVERGENCE_NO_CHANGE_FOR][order_i] = network.forwardPass(in)[2]; 
			}

			//no weight change
			avg_weight_change /= (double) (order.size()); 
			if(avg_weight_change < BAL.CONVERGENCE_WEIGHT_EPSILON){
				log.println("Training stopped at epoch=" + epochs + " with avg_weight_change=" + avg_weight_change);
				break;
			}

			//no output change
			boolean output_change = true; 
			if(epochs >= CONVERGENCE_NO_CHANGE_FOR){
				int a = epochs % CONVERGENCE_NO_CHANGE_FOR;
				int b = (epochs + 1) % CONVERGENCE_NO_CHANGE_FOR; 
				double max_diff = 0.0;

				for(int j=0; j<given[0].length; j++){
					for(int k=0; k<given[0][0].getDimension() ; k++){
						max_diff = Math.max(max_diff, Math.abs(given[a][j].getEntry(k) - given[b][j].getEntry(k)));
					}
				}
				
				output_change = (max_diff > CONVERGENCE_NO_CHANGE_EPSILON); 
				//log.println("  max_diff=" + max_diff);
			}
			if(!output_change){
				log.println("Training stopped at epoch=" + epochs + " as no output change occured in last " + CONVERGENCE_NO_CHANGE_FOR + "epochs");
				break;
			}
		}

		double network_result = network.evaluate(inputs, outputs);
		
		//print only "bad results" 
		if(network_result > 0.0){
			if(PRINT_NETWORK_IS){
				log.println(network.printNetwork());
			}
	
			//print each input output activations 
			for(int i=0 ; i<inputs.getRowDimension(); i++){
				RealVector[] forward = network.forwardPass(inputs.getRowVector(i));
	
				if(PRINT_NETWORK_IS){
					log.println("Forward pass:");
					for(int j=0; j<forward.length; j++){
						log.print(BAL.printVector(forward[j]));
					}
				}
	
				network.postprocessOutput(forward[2]);
				log.print("Given:   " + BAL.printVector(forward[2]));
				log.print("Expected:" + BAL.printVector(outputs.getRowVector(i)));
				log.println();
			}
			//print each input output activations 
			if(PRINT_NETWORK_IS){
				for(int i=0 ; i<outputs.getRowDimension(); i++){
					RealVector[] backward = network.backwardPass(outputs.getRowVector(i));
	
					log.println("Backward pass:");
					for(int j=0; j<3; j++){
						log.print(BAL.printVector(backward[j]));
					}
				}
			}
		}

		if(MEASURE_IS) {
			post_measure.add(network.measure(epochs, inputs, outputs));
		}
		//log.println(network.printNetwork());

		if(BAL.MEASURE_IS && BAL.MEASURE_SAVE_AFTER_EACH_RUN){
			network.saveMeasures("data/measure_" + ((int)network.evaluate(inputs, outputs)) + "_" + (System.currentTimeMillis() / 1000L) + ".dat");
		}

		log.println("Result=" + network_result);
		log.println("=========================================================");
		System.out.println("Epochs=" + epochs);
		System.out.println("Result=" + network_result);
		System.out.println("=========================================================");
	}

	//interpret activations on the output layer 
	//for example map continuous [0,1] data to discrete {0, 1} 
	public void postprocessOutput(RealVector out){

		//normal sigmoid 
		//[0.5,\=+\infty] -> 1.0, else 0.0
		for(int i=0; i<out.getDimension() ;i++){
			out.setEntry(i, (out.getEntry(i) >= 0.50) ? 1.0 : 0.0); 
		}

		/*
		//bipolar sigmoid 
		//[0.0,\=+\infty] -> 1.0, else 0.0
		for(int i=0; i<out.getDimension() ;i++){
			out.setEntry(i, (out.getEntry(i) >= 0.00) ? 1.0 : 0.0); 
		}*/

		/*
		//maximum
		int m_i = 0;
		for(int i=0; i<out.getDimension() ;i++){
			if(out.getEntry(i) > out.getEntry(m_i)){
				m_i = i; 
			}
		}
		for(int i=0; i<out.getDimension() ;i++){
			out.setEntry(i, 0.0); 
		}
		out.setEntry(m_i, 1.0);
		 */ 
	}

	//computes \phi((x - mu)/sigma)) 
	public static double normalDistributionValue(double mu, double sigma, double x){
		return (1 / (Math.sqrt(2*Math.PI*sigma*sigma))) * Math.exp(- ((Math.pow(x - mu, 2))/(2*sigma*sigma)));
	}

	//choses randomly from NormalDistribution from interval [-NORMAL_DISTRIBUTION_SPAN, NORMAL_DISTRIBUTION_SPAN] 
	public static double pickFromNormalDistribution(double mu, double sigma){
		double ma = normalDistributionValue(mu, sigma, 0); 
		while(true){
			double x = (2*Math.random() - 1)*NORMAL_DISTRIBUTION_SPAN; 
			double y = Math.random()*ma;
			if(y <= normalDistributionValue(mu, sigma, x)) {
				return x;
			}
		}
	}

	//Creates a random weight matrix 
	private RealMatrix createInitMatrix(int rows, int cols){
		double [][] matrix_data = new double[rows][cols]; 
		for(int i=0; i<rows; i++){
			for(int j=0; j<cols; j++){
				matrix_data[i][j] = pickFromNormalDistribution(BAL.INIT_NORMAL_DISTRIBUTION_MU, BAL.INIT_NORMAL_DISTRIBUTION_SIGMA);
				//matrix_data[i][j] = Math.random()*5 - 2.5; 
			}
		}

		RealMatrix m = MatrixUtils.createRealMatrix(matrix_data);
		return m; 
	}

	//Creates a BAL network with layer sizes [in_size, h_size, out_size] 
	public BAL(int in_size, int h_size, int out_size) {
		//+1 stands for biases 
		//we use matrix premultiply and vertical vectors A*v 
		this.IH = createInitMatrix(in_size+1, h_size);
		this.HO = createInitMatrix(h_size+1, out_size);
		this.OH = createInitMatrix(out_size+1, h_size);
		this.HI = createInitMatrix(h_size+1, in_size);

		this.MOM_IH = new double[in_size+1][h_size];
		this.MOM_HO = new double[h_size+1][out_size];
		this.MOM_OH = new double[out_size+1][h_size];
		this.MOM_HI = new double[h_size+1][in_size];

		this.measures = new ArrayList[MEASURE_COUNT]; 
		for(int i=0; i<MEASURE_COUNT; i++){
			this.measures[i] = new ArrayList<Double>();
		}
	}

	//TODO consider k*n in exponent 
	//f(net) on a whole layer  
	private void applyNonlinearity(RealVector vector){
		for(int i=0; i<vector.getDimension() ; i++){
			double n = vector.getEntry(i); 
			vector.setEntry(i, 1.0 / (1.0 + Math.exp(-n)));  //normal sigmoid 
			//vector.setEntry(i, 1 - (2 / (1 + Math.exp(-n)))); //bipolar sigmoid 
		}
	}

	private RealVector addBias(RealVector in){
		return in.append(1); 
	}

	//forward activations 
	private RealVector[] forwardPass(RealVector in){
		RealVector[] forward = new RealVector[3]; 
		forward[0] = addBias(in);

		forward[1] = this.IH.preMultiply(forward[0]);
		applyNonlinearity(forward[1]);
		forward[1] = addBias(forward[1]); 

		forward[2] = this.HO.preMultiply(forward[1]);
		applyNonlinearity(forward[2]);

		return forward; 
	}

	//backward activations 
	private RealVector[] backwardPass(RealVector out){
		RealVector[] backward = new RealVector[3]; 
		backward[2] = addBias(out); 

		backward[1] = this.OH.preMultiply(backward[2]);
		applyNonlinearity(backward[1]);
		backward[1] = addBias(backward[1]); 

		backward[0] = this.HI.preMultiply(backward[1]);
		applyNonlinearity(backward[0]);

		return backward; 
	}

	//learns on a weight matrix, other parameters are activations on needed layers 
	//\delta w_{pq}^F = \lambda a_p^{F}(a_q^{B} - a_q^{F})
	//\delta w_ij = lamda * a_pre * (a_post_other - a_post_self)  
	private double subLearn(RealMatrix w, RealVector a_pre, RealVector a_post_other, RealVector a_post_self, double lambda, double[][] mom){
		double avg_change = 0.0; 

		for(int i = 0 ; i < w.getRowDimension() ; i++){
			for(int j = 0 ; j < w.getColumnDimension() ; j++){
				double w_value = w.getEntry(i, j); 
				double dw = lambda * a_pre.getEntry(i) * (a_post_other.getEntry(j) - a_post_self.getEntry(j));
				w.setEntry(i, j, w_value + dw + BAL.INIT_MOMENTUM * mom[i][j]);

				mom[i][j] = dw; 
				avg_change += Math.abs(dw / w_value); 
			}
		}

		return avg_change / ((double)(w.getRowDimension() * w.getColumnDimension()));
	}

	//learn on one input-output mapping
	// returns avg weight change 
	public double learn(RealVector in, RealVector target, double lambda){
		//forward and backward activation
		RealVector[] forward = this.forwardPass(in);
		RealVector[] backward = this.backwardPass(target);
		double avg_change_ih = 0.0; 
		double avg_change_oh = 0.0; 

		//learn 
		avg_change_ih += subLearn(this.IH, forward[0], backward[1], forward[1], lambda, this.MOM_IH); 
		avg_change_oh += subLearn(this.HO, forward[1], backward[2], forward[2], lambda, this.MOM_HO); 
		avg_change_ih += subLearn(this.OH, backward[2], forward[1], backward[1], lambda, this.MOM_OH); 
		avg_change_oh += subLearn(this.HI, backward[1], forward[0], backward[0], lambda, this.MOM_HI); 

		//this.addNoise(this.OH, BAL.INIT_MULTIPLY_WEIGHTS);
		//this.addNoise(this.HI, BAL.INIT_MULTIPLY_WEIGHTS); 

		/*
		log.println("Forward pass:");
		for(int i=0; i<3; i++){
			log.print(BAL.printVector(forward[i]));
		}
		log.println("Backward pass:");
		for(int i=0; i<3; i++){
			log.print(BAL.printVector(backward[2-i]));
		}*/
		//log.print(BAL.printVector(forward[1]));
		//log.println(BAL.printVector(backward[1]));

		double size_ih = this.IH.getColumnDimension() * this.IH.getRowDimension();
		double size_oh = this.OH.getColumnDimension() * this.OH.getRowDimension(); 
		return (avg_change_ih * size_ih + avg_change_oh * size_oh) / (size_ih + size_oh); 
	}

	//evaluates performance on one input-output mapping 
	//returns error 
	public double evaluate(RealVector in, RealVector target){
		RealVector[] forward = forwardPass(in);
		RealVector result = forward[forward.length - 1]; 
		this.postprocessOutput(result);

		double error = 0.0; 

		for(int i=0; i<target.getDimension() ; i++){
			error += Math.pow(result.getEntry(i) - target.getEntry(i), 2); 
		}

		return error; 	
	}

	//evaluates performance on several input-output mapping 
	//returns absolute error 
	public double evaluate(RealMatrix in, RealMatrix target){
		double error = 0.0; 
		for(int i=0; i<in.getRowDimension() ; i++){
			error += this.evaluate(in.getRowVector(i), target.getRowVector(i)); 
		}
		return error; 
	}

	public static double sumAbsoluteValuesOfMatrixEntries(RealMatrix m){
		double sum = 0.0; 
		for(int i = 0 ; i < m.getRowDimension() ; i++){
			for(int j = 0 ; j < m.getColumnDimension() ; j++){
				sum += Math.abs(m.getEntry(i, j)); 
			}
		}
		return sum; 
	}

	//collect monitoring data, epoch is used as identifier
	//  !this data is also stored into measures array 
	public double[] measure(int epoch, RealMatrix in, RealMatrix target){
		double n = in.getRowDimension(); 

		double hidden_dist = 0.0;
		double hidden_for_back_dist = 0.0;
		double output_for_back_dist = 0.0;
		double matrix_avg_w = 0.0;
		double matrix_similarity = 0.0;

		this.measures[MEASURE_EPOCH].add((double)epoch); 
		this.measures[MEASURE_SIGMA].add(BAL.INIT_NORMAL_DISTRIBUTION_SIGMA); 
		this.measures[MEASURE_LAMBDA].add(BAL.INIT_LAMBDA); 
		this.measures[MEASURE_MOMENTUM].add(BAL.INIT_MOMENTUM); 
		//this.measures[MEASURE_NOISE_SPAN].add(BAL.INIT_NOISE_SPAN); 
		//this.measures[MEASURE_MULTIPLY_WEIGHTS].add(BAL.INIT_MULTIPLY_WEIGHTS - 1); 

		this.measures[MEASURE_ERROR].add(this.evaluate(in, target)); 

		ArrayList<RealVector> forward_hiddens = new ArrayList<RealVector>(); 

		double first_second_sum = 0.0; 
		for(int i=0; i<in.getRowDimension(); i++){
			RealVector[] forward = this.forwardPass(in.getRowVector(i));
			RealVector[] backward = this.backwardPass(target.getRowVector(i));

			hidden_for_back_dist += forward[1].getDistance(backward[1]) / n; 
			output_for_back_dist += forward[2].getDistance(backward[0]) / n; 

			forward_hiddens.add(forward[1]);

			double[] output_arr = forward[2].toArray();
			if(output_arr.length > 1){
				Arrays.sort(output_arr); 
				first_second_sum += output_arr[output_arr.length-1] / output_arr[output_arr.length-2];
			}
		}
		this.measures[MEASURE_HIDDEN_FOR_BACK_DIST].add(hidden_for_back_dist); 
		this.measures[MEASURE_OUTPUT_FOR_BACK_DIST].add(output_for_back_dist); 
		this.measures[MEASURE_FIRST_SECOND_RATIO].add(first_second_sum); 

		for(int i=0; i<forward_hiddens.size() ; i++){
			for(int j=i+1; j<forward_hiddens.size() ; j++){
				hidden_dist += forward_hiddens.get(i).getDistance(forward_hiddens.get(j)) / (forward_hiddens.size() * (forward_hiddens.size() + 1) / 2); 
			}
		}

		this.measures[MEASURE_HIDDEN_DIST].add(hidden_dist);  

		matrix_avg_w = (sumAbsoluteValuesOfMatrixEntries(this.IH) + sumAbsoluteValuesOfMatrixEntries(this.HO) + sumAbsoluteValuesOfMatrixEntries(this.OH) + sumAbsoluteValuesOfMatrixEntries(this.IH)) / (this.IH.getColumnDimension()*this.IH.getRowDimension() + this.HO.getColumnDimension()*this.HO.getRowDimension()+ this.OH.getColumnDimension()*this.OH.getRowDimension()+ this.HI.getColumnDimension()*this.HI.getRowDimension()); 
		this.measures[MEASURE_MATRIX_AVG_W].add(matrix_avg_w);   

		if(MEASURE_MATRIX_SIMILARITY >= 0 && this.HO.getColumnDimension() == this.HI.getColumnDimension() && this.HO.getRowDimension() == this.HI.getRowDimension()){
			RealMatrix diff_HO_HI = this.HO.subtract(this.HI); 
			RealMatrix diff_OH_IH = this.OH.subtract(this.IH);
			matrix_similarity = (sumAbsoluteValuesOfMatrixEntries(diff_HO_HI) + sumAbsoluteValuesOfMatrixEntries(diff_OH_IH)) / (this.IH.getColumnDimension()*this.IH.getRowDimension() + this.HI.getColumnDimension()*this.HI.getRowDimension());   
			this.measures[MEASURE_MATRIX_SIMILARITY].add(matrix_similarity);
		}

		double[] result = new double[MEASURE_COUNT]; 
		for(int i=0; i<MEASURE_COUNT; i++){
			result[i] = this.measures[i].get(this.measures[i].size()-1); 
		}
		return result; 
	}

	private static void measureGroupBY(ArrayList<double[]> measures) {
		int m = BAL.MEASURE_GROUP_BY_COLS.length; 
		//List<Map<Double, Integer>> group_ids = new ArrayList<Map<Double,Integer>>();
		//_{group1_id}_..._{groupn_id} -> count  
		Map<String, Integer> counts_child = new HashMap<String, Integer>(); 
		Map<String, Integer> counts_parent = new HashMap<String, Integer>(); 
		Map<String, String> child2parent = new HashMap<String, String>(); 

		for(int i=0; i<measures.size(); i++){
			String s_child = "";
			String s_parent = ""; 
			for(int j=0; j<m; j++){
				int id = BAL.MEASURE_GROUP_BY_COLS[j]; 
				double val = measures.get(i)[id];

				s_child += val + " ";
				if(id != BAL.MEASURE_GROUP_BY){
					s_parent += val + " "; 
				}
			}
			child2parent.put(s_child, s_parent);

			if(!counts_child.containsKey(s_child)) {
				counts_child.put(s_child, 0);
			}
			if(!counts_parent.containsKey(s_parent)) {
				counts_parent.put(s_parent, 0);
			}
			counts_child.put(s_child, counts_child.get(s_child) + 1); 
			counts_parent.put(s_parent, counts_parent.get(s_parent) + 1);
		}

		for(int j=0; j<m; j++){
			if(j != 0) log.print(" ");
			log.print(BAL.MEASURE_HEADINGS[BAL.MEASURE_GROUP_BY_COLS[j]]);
		}
		log.println(" success sample_ratio");

		List<String> result = new ArrayList<String>(); 
		for(Entry<String, Integer> entry : counts_child.entrySet()){
			Integer child_count = entry.getValue();
			Integer parent_count = counts_parent.get(child2parent.get(entry.getKey())); 
			result.add(entry.getKey() + (100.0*((double)child_count / (double)parent_count)) + " " + child_count + "/" + parent_count);
		}
		Collections.sort(result); 
		for(String s : result){
			log.println(s);
		}
	}

	private static void measureAverages(ArrayList<double[]> measures) {

		double[] sum = new double[measures.get(0).length]; 
		for(int i=0; i<measures.size() ; i++){
			for(int j=0; j<measures.get(i).length ; j++){
				sum[j] += measures.get(i)[j]; 
			}
		}

		for(int j=0; j<measures.get(0).length ; j++){
			log.println("avg("+BAL.MEASURE_HEADINGS[j]+")=" + (sum[j] / ((double)measures.size())));
		}
	}

	public static RealMatrix loadFromFile(String filepath){
		Scanner in = null; 
		try {
			in = new Scanner(new FileReader(filepath));
		} catch (FileNotFoundException e) {
			return null; 
		}

		int rows = in.nextInt();
		int cols = in.nextInt();
		double[][] data = new double[rows][cols]; 

		for(int i=0; i<rows; i++){
			for(int j=0; j<cols; j++){
				data[i][j] = in.nextDouble(); 
			}
		}

		return MatrixUtils.createRealMatrix(data); 
	}

	public static String printVector(RealVector v){
		StringBuilder sb = new StringBuilder();
		//sb.append(v.getDimension());
		//sb.append('\n'); 

		for(int i = 0 ; i < v.getDimension() ; i++){
			if(i!=0){
				sb.append(' ');
			}
			sb.append(v.getEntry(i));
		}
		sb.append('\n');

		return sb.toString(); 
	}

	public static String printMatrix(RealMatrix m){
		StringBuilder sb = new StringBuilder();
		sb.append(m.getRowDimension());
		sb.append(' '); 
		sb.append(m.getColumnDimension());
		sb.append('\n'); 

		for(int i = 0 ; i < m.getRowDimension() ; i++){
			for(int j = 0 ; j < m.getColumnDimension() ; j++){
				if(j!=0){
					sb.append(' ');
				}
				sb.append(m.getEntry(i, j));
			}
			sb.append('\n');
		}
		return sb.toString(); 
	}

	public String printNetwork(){
		StringBuilder sb = new StringBuilder(); 
		sb.append("BAL network of size " + (this.IH.getRowDimension()-1) + "," + this.IH.getColumnDimension() + "," + this.HO.getColumnDimension() + "\n");
		sb.append("IH\n"); 
		sb.append(BAL.printMatrix(this.IH));
		sb.append("HO\n"); 
		sb.append(BAL.printMatrix(this.HO));
		sb.append("OH\n"); 
		sb.append(BAL.printMatrix(this.OH));
		sb.append("HI\n"); 
		sb.append(BAL.printMatrix(this.HI));
		return sb.toString(); 
	}

	//TODO Refactor with printPreAndPostMeasures()
	//saves this.measures into file 
	public boolean saveMeasures(String filename){
		if(!MEASURE_IS) return true; 

		PrintWriter writer;
		try {
			writer = new PrintWriter(filename, "UTF-8");
		} catch (Exception e) {
			return false; 
		} 

		double[] m = new double[MEASURE_COUNT];
		for(int j=0; j<MEASURE_COUNT; j++){
			m[j] = 1;
		}

		//normalize all values to [0,1] 
		/*
			for(int i=0; i<this.measures[0].size(); i++){
				for(int j=0; j<this.measures.length; j++){
					m[j] = Math.max(m[j], this.measures[j].get(i));
				}
			}
			m[MEASURE_EPOCH] = 1;
		 */ 

		//writer.println(this.measures[0].size() + " " + this.measures.length); 
		for(int i=0; i<MEASURE_HEADINGS.length ; i++){
			if(i != 0){
				writer.write(' ');
			}
			writer.write(MEASURE_HEADINGS[i]); 
		}
		writer.println(); 

		for(int i=0; i<this.measures[0].size(); i++){
			for(int j=0; j<this.measures.length; j++){
				if(j != 0){
					writer.print(' ');
				}
				writer.print(this.measures[j].get(i) / m[j]); 
			}
			writer.println(); 
		}

		writer.close();

		return true; 
	}	

	//TODO Refactor with saveMeasures()
	public static void printPreAndPostMeasures(){
		if(!MEASURE_IS) return; 

		PrintWriter pre_writer;
		PrintWriter post_writer;
		try {
			pre_writer = new PrintWriter("data/pre_measure_" + RUN_ID + ".dat", "UTF-8");
			post_writer = new PrintWriter("data/post_measure_" + RUN_ID + ".dat", "UTF-8");
		} catch (Exception e) {
			return; 
		} 

		for(int i=0; i<MEASURE_HEADINGS.length ; i++){
			if(i != 0){
				pre_writer.write(' ');
				post_writer.write(' ');
			}
			pre_writer.write(MEASURE_HEADINGS[i]); 
			post_writer.write(MEASURE_HEADINGS[i]); 
		}
		pre_writer.println();
		post_writer.println();

		for(int i=0; i<pre_measure.size() ; i++){
			pre_measure.get(i)[MEASURE_ERROR] = post_measure.get(i)[MEASURE_ERROR];
			for(int j=0; j<MEASURE_COUNT ; j++){
				if(j != 0){
					pre_writer.write(' ');
					post_writer.write(' ');
				}
				pre_writer.print(pre_measure.get(i)[j]); 
				post_writer.print(post_measure.get(i)[j]); 
			}
			pre_writer.println();
			post_writer.println();
		}

		pre_writer.close();
		post_writer.close();

		BAL.measureGroupBY(pre_measure); 
		BAL.measureAverages(post_measure); 
	}

	//manage IO and run BAL 
	public static void main(String[] args) throws FileNotFoundException {
		for(int h=3; h<17; h++){
			initMultidimensional("k3", h);
			experiment();
		}
	}
	
	public static void experiment() throws FileNotFoundException{
		RUN_ID = (System.currentTimeMillis() / 1000L) + "_" + INIT_HIDDEN_LAYER_SIZE;
		
		String filename = "data/" + RUN_ID + ".log"; 
		log = new PrintWriter(filename);
		
		for(int i=0 ; i<BAL.INIT_RUNS ; i++){
			log.println("======== " + i + "/" + BAL.INIT_RUNS + " ==============");
			System.out.println("======== " + i + "/" + BAL.INIT_RUNS + " ==============");
			BAL.run(); 
		}

		printPreAndPostMeasures();
		log.close(); 
	}

	public static void initMultidimensional(String input_prefix, int hidden_size){
		MEASURE_IS = true; 
		MEASURE_SAVE_AFTER_EACH_RUN = false; 
		MEASURE_RECORD_EACH = 1000;

		INPUT_FILEPATH = input_prefix + ".in"; 
		OUTPUT_FILEPATH = input_prefix + ".out"; 
		INIT_HIDDEN_LAYER_SIZE = hidden_size; 

		CONVERGENCE_WEIGHT_EPSILON = 0.0; 
		
		CONVERGENCE_NO_CHANGE_FOR = 10; 
		CONVERGENCE_NO_CHANGE_EPSILON = 0.001;
		INIT_MAX_EPOCHS = 5000;

		INIT_RUNS = 1000; 
		INIT_CANDIDATES_COUNT = 1;

		PRINT_NETWORK_IS = true; 

		TRY_NORMAL_DISTRIBUTION_SIGMA = new double[] {2.3}; 
		TRY_LAMBDA = new double[] {0.7}; 
		TRY_MOMENTUM = new double[] {0.0}; 
	}
}
