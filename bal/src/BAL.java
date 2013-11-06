//TODO kvazi momentum -> odtlacanie hidden reprezentacii
//TODO m_sim, h_f_b_dist -> jeden nasobkom druheho
//TODO bipolarna sigmoida (preferujeme unipolarnu)
//TODO dropout? 
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
import java.util.Set;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class BAL {
	private RealMatrix IH; 
	private RealMatrix HO;  
	private RealMatrix OH; 
	private RealMatrix HI;

	public static final String[] MEASURE_HEADINGS = {"epoch","err","h_dist","h_f_b_dist","m_avg_w","m_sim", "first_second", "sigma", "lambda"}; 
	
	//TODO activation on hidden networks 
	//epoch
	public static final int MEASURE_EPOCH = 0; 
	
	//error function (RMSE) 
	public static final int MEASURE_ERROR = 1;
	
	//avg of dist(h_i - h_j) i \neq j where h_i is a hidden activation for input i
	//intuitively: internal representation difference 
	public static final int MEASURE_HIDDEN_DIST = 2;
	
	//avg distance between forward and backward activations on hidden layer
	public static final int MEASURE_HIDDEN_FOR_BACK_DIST = 3;
	
	//avg weight of matrixes 
	public static final int MEASURE_MATRIX_AVG_W = 4;
	
	//sum of |a_{ij} - b_{ij}| per pairs (HO, HI) and (OH, IH) 
	public static final int MEASURE_MATRIX_SIMILARITY = 5;
	
	//ratio of (a_1, a_2) where a_i is the i-th biggest output 
	public static final int MEASURE_FIRST_SECOND_RATIO = 6;
	
	public static final int MEASURE_SIGMA = 7; 
	public static final int MEASURE_LAMBDA = 8; 
	//public static final int MEASURE_NOISE_SPAN = 9; 
	//public static final int MEASURE_MULTIPLY_WEIGHTS = 9; 
	
	public static final int MEASURE_COUNT = 9; 
	
	public static final int[] MEASURE_GROUP_BY_COLS = {MEASURE_ERROR, MEASURE_SIGMA, MEASURE_LAMBDA};
	public static final int MEASURE_GROUP_BY = MEASURE_ERROR;  
	
	public static final double NORMAL_DISTRIBUTION_SPAN = 15; 

	public static final String INPUT_FILEPATH = "auto4.in"; 
	public static final String OUTPUT_FILEPATH = "auto4.in"; 
	public static final int INIT_HIDDEN_LAYER_SIZE = 2; 
	public static final int INIT_MAX_EPOCHS = 30000;
	public static final int INIT_RUNS = 1000; 
	public static final double INIT_NORMAL_DISTRIBUTION_MU = 0; 
	public static final double TRY_NORMAL_DISTRIBUTION_SIGMA[] = {2.3}; 
	//public static final double TRY_NORMAL_DISTRIBUTION_SIGMA[] = {1.5, 1.7, 1.9, 2.1, 2.3, 2.5}; 
	public static final double TRY_LAMBDA[] = {0.7}; 
	//public static final double TRY_LAMBDA[] = {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}; 
	//public static final double TRY_LAMBDA[] = {0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5}; 
	//public static final double TRY_NOISE_SPAN[] = {0.0, 0.003, 0.01, 0.03, 0.1, 0.3}; 
	//public static final double TRY_MULTIPLY_WEIGHTS[] = {1.0, 1.00001, 1.00003, 1.0001, 1.0003, 1.001}; 
	
	public static Random random = new Random(); 
	public static double INIT_NORMAL_DISTRIBUTION_SIGMA = 1.25; 
	public static double INIT_LAMBDA = 0.03; 
	//public static double INIT_NOISE_SPAN = 0.00; 
	//public static double INIT_MULTIPLY_WEIGHTS = 1.001; 

	public static final int MEASURE_RECORD_EACH = 1000; 
	public static boolean MEASURE_SAVE_AFTER_EACH_RUN = false; 
	
	private ArrayList<Double>[] measures = null; 

	public static ArrayList<double[]> pre_measure = new ArrayList<double[]>();
	public static ArrayList<double[]> post_measure = new ArrayList<double[]>();
	
	public static void run(){
		BAL.INIT_NORMAL_DISTRIBUTION_SIGMA = BAL.TRY_NORMAL_DISTRIBUTION_SIGMA[random.nextInt(BAL.TRY_NORMAL_DISTRIBUTION_SIGMA.length)]; 
		BAL.INIT_LAMBDA = BAL.TRY_LAMBDA[random.nextInt(BAL.TRY_LAMBDA.length)];
		//BAL.INIT_NOISE_SPAN = BAL.TRY_NOISE_SPAN[random.nextInt(BAL.TRY_NOISE_SPAN.length)];
		//BAL.INIT_MULTIPLY_WEIGHTS = BAL.TRY_MULTIPLY_WEIGHTS[random.nextInt(BAL.TRY_MULTIPLY_WEIGHTS.length)];
		
		int h_size = BAL.INIT_HIDDEN_LAYER_SIZE; 
		double lambda = BAL.INIT_LAMBDA; 
		int max_epoch = BAL.INIT_MAX_EPOCHS; 
		
		RealMatrix inputs = BAL.loadFromFile(BAL.INPUT_FILEPATH);
		RealMatrix outputs = BAL.loadFromFile(BAL.OUTPUT_FILEPATH);
		
		BAL network = new BAL(inputs.getColumnDimension(), h_size, outputs.getColumnDimension()); 
		//System.out.println(network.printNetwork()); 
		
		ArrayList<Integer> order = new ArrayList<Integer>(inputs.getRowDimension());
		for(int i=0; i<inputs.getRowDimension() ; i++){
			order.add(i); 
		}

		pre_measure.add(network.measure(0, inputs, outputs));
		
		for(int e=0; e<max_epoch ; e++){
			if(MEASURE_SAVE_AFTER_EACH_RUN && e % BAL.MEASURE_RECORD_EACH == 0){
				//System.out.println(network.evaluate(inputs, outputs));
				network.measure(e, inputs, outputs);
			}
			
			java.util.Collections.shuffle(order);
			for(int order_i = 0; order_i < order.size() ; order_i++){
				RealVector in = inputs.getRowVector(order_i);
				RealVector out = outputs.getRowVector(order_i);
				network.learn(in, out, lambda); 
			}
		}
		 
		System.out.println(network.printNetwork());
		//print each input output activations 
		for(int i=0 ; i<4; i++){
			RealVector[] forward = network.forwardPass(inputs.getRowVector(i));
			RealVector[] backward = network.backwardPass(inputs.getRowVector(i));
			
			System.out.println("Forward pass:");
			for(int j=0; j<forward.length; j++){
				System.out.print(BAL.printVector(forward[j]));
			}
			
			network.postprocessOutput(forward[2]);
			System.out.print("Given:   " + BAL.printVector(forward[2]));
			System.out.print("Expected:" + BAL.printVector(outputs.getRowVector(i)));
			
			
			/*
			System.out.println("Backward pass:");
			for(int j=0; j<3; j++){
				System.out.print(BAL.printVector(backward[j]));
			}*/
		}
		
		System.out.println(network.evaluate(inputs, outputs));
		System.out.println("=========================================================");
		
		post_measure.add(network.measure(max_epoch, inputs, outputs));
		//System.out.println(network.printNetwork());
		
		if(BAL.MEASURE_SAVE_AFTER_EACH_RUN){
			network.saveMeasures("data/measure_" + ((int)network.evaluate(inputs, outputs)) + "_" + (System.currentTimeMillis() / 1000L) + ".dat");
		}
	}
	
	//interpret activations on the output layer 
	//for example map continuous [0,1] data to discrete {0, 1} 
	public void postprocessOutput(RealVector out){
		//[0.5,\=+\infty] -> 1.0, else 0.0
		for(int i=0; i<out.getDimension() ;i++){
			out.setEntry(i, (out.getEntry(i) >= 0.50) ? 1.0 : 0.0); 
		}
		
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
			vector.setEntry(i, 1 / (1 + Math.exp(-n)));
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
	private void subLearn(RealMatrix w, RealVector a_pre, RealVector a_post_other, RealVector a_post_self, double lambda){
		for(int i = 0 ; i < w.getRowDimension() ; i++){
			for(int j = 0 ; j < w.getColumnDimension() ; j++){
				double w_value = w.getEntry(i, j); 
				double dw = lambda * a_pre.getEntry(i) * (a_post_other.getEntry(j) - a_post_self.getEntry(j));
				if(dw != 0){
					w.setEntry(i, j, w_value + dw);
				}
			}
		}
	}
	
	//learn on one input-output mapping 
	public void learn(RealVector in, RealVector target, double lambda){
		//forward and backward activation
		RealVector[] forward = this.forwardPass(in);
		RealVector[] backward = this.backwardPass(target);
		
		//learn 
		subLearn(this.IH, forward[0], backward[1], forward[1], lambda); 
		subLearn(this.HO, forward[1], backward[2], forward[2], lambda); 
		subLearn(this.OH, backward[2], forward[1], backward[1], lambda); 
		subLearn(this.HI, backward[1], forward[0], backward[0], lambda); 
		
		//this.addNoise(this.OH, BAL.INIT_MULTIPLY_WEIGHTS);
		//this.addNoise(this.HI, BAL.INIT_MULTIPLY_WEIGHTS); 
		
		/*
		System.out.println("Forward pass:");
		for(int i=0; i<3; i++){
			System.out.print(BAL.printVector(forward[i]));
		}
		System.out.println("Backward pass:");
		for(int i=0; i<3; i++){
			System.out.print(BAL.printVector(backward[2-i]));
		}*/
		//System.out.print(BAL.printVector(forward[1]));
		//System.out.println(BAL.printVector(backward[1]));
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
		double for_back_dist = 0.0;
		double matrix_avg_w = 0.0;
		double matrix_similarity = 0.0;

		this.measures[MEASURE_EPOCH].add((double)epoch); 
		this.measures[MEASURE_SIGMA].add(BAL.INIT_NORMAL_DISTRIBUTION_SIGMA); 
		this.measures[MEASURE_LAMBDA].add(BAL.INIT_LAMBDA); 
		//this.measures[MEASURE_NOISE_SPAN].add(BAL.INIT_NOISE_SPAN); 
		//this.measures[MEASURE_MULTIPLY_WEIGHTS].add(BAL.INIT_MULTIPLY_WEIGHTS - 1); 
		
		this.measures[MEASURE_ERROR].add(this.evaluate(in, target)); 
		
		ArrayList<RealVector> forward_hiddens = new ArrayList<RealVector>(); 
		
		double first_second_sum = 0.0; 
		for(int i=0; i<in.getRowDimension(); i++){
			RealVector[] forward = this.forwardPass(in.getRowVector(i));
			RealVector[] backward = this.backwardPass(target.getRowVector(i));
			
			for_back_dist += forward[1].getDistance(backward[1]) / n; 
			forward_hiddens.add(forward[1]);
			
			double[] output_arr = forward[2].toArray();
			if(output_arr.length > 1){
				Arrays.sort(output_arr); 
				first_second_sum += output_arr[output_arr.length-1] / output_arr[output_arr.length-2];
			}
		}
		this.measures[MEASURE_HIDDEN_FOR_BACK_DIST].add(for_back_dist); 
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
			System.out.print(" " + BAL.MEASURE_HEADINGS[BAL.MEASURE_GROUP_BY_COLS[j]]);
		}
		System.out.println();
		
		List<String> result = new ArrayList<String>(); 
		for(Entry<String, Integer> entry : counts_child.entrySet()){
			Integer child_count = entry.getValue();
			Integer parent_count = counts_parent.get(child2parent.get(entry.getKey())); 
			result.add(entry.getKey() + child_count + "/" + parent_count + " " + 100.0*((double)child_count / (double)parent_count) + "%");
		}
		Collections.sort(result); 
		for(String s : result){
			System.out.println(s);
		}
	}
	
	//saves this.measures into file 
	public boolean saveMeasures(String filename){
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

	//manage IO and run BAL 
	public static void main(String[] args) {
		for(int i=0 ; i<BAL.INIT_RUNS ; i++){
			BAL.run(); 
		}

		PrintWriter pre_writer;
		PrintWriter post_writer;
		try {
			pre_writer = new PrintWriter("data/pre_measure_" + (System.currentTimeMillis() / 1000L) + ".dat", "UTF-8");
			post_writer = new PrintWriter("data/post_measure_" + (System.currentTimeMillis() / 1000L) + ".dat", "UTF-8");
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
		
		BAL.measureGroupBY(post_measure); 
	}
		
}
