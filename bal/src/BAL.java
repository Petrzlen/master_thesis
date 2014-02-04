//TODO O'Really - kedy podobny backpropagation 
//  -- ci sa naozaj snazi minimalizovat gradient 

//TODO Refactor 

//TODO ==> ako rozlisit uspesne a neuspesne od inicializacie (v batch mode)
//TODO binarny klasifikator na good/bad vah 

//TODO graf od poctu epoch

//TODO v result vypisat aj pocet epoch
//TODO pridat deliace ciary 
//TODO porovnat vahove matice pri viacnasobnom behu rovnakej siete
//TODO inicializovat protilahle matice zavisle na sebe 

//TODO zovseobecnenie siete

//TODO measure min, max distance from target 
//TODO dropout? 
//TODO matematicky pohlad - o'really clanok aproximacia gradientu chyby
//TODO rekonstrukcia (zmena zopar bitov, ci tam-speat da orig) 
//TODO vyssi rozmer tasku (8-3-8), (16-4-16)
//TODO kvazi momentum -> odtlacanie hidden reprezentacii, -\delta w(t-1) 
//TODO pocet epoch potrebnych na konvergenciu
//TODO nie autoassoc ale permutovat vystupy (napr. 1000 na 0100)
//TODO reprezentacia SUC/ERR 
//TODO m_sim, h_f_b_dist -> jeden nasobkom druheho bias
//TODO spektralna analyza
//TODO bipolarna [-1, 1] vstupy
//h_size = 3 => [9,10,1] for errors [0.0, 1.0, 2.0] 

import java.awt.Point;
import java.awt.Polygon; 
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class BAL {
	private static final boolean SYMMETRIC_WEIGHT_UPDATE = false;

	private static PrintWriter log = null; 

	public static double DOUBLE_EPSILON = 0.001;
	
	public static  boolean MEASURE_IS = true; 
	public static boolean MEASURE_SAVE_AFTER_EACH_RUN = false; 
	public static  int MEASURE_RECORD_EACH = 1000;

	public static  String INPUT_FILEPATH = "auto4.in"; 
	public static  String OUTPUT_FILEPATH = "auto4.in"; 
	public static  int INIT_HIDDEN_LAYER_SIZE = 2 ; 

	public static  double CONVERGENCE_WEIGHT_EPSILON = 0.0; 
	//there was no change in given outputs for last CONVERGENCE_NO_CHANGE_FOR
	public static  int CONVERGENCE_NO_CHANGE_FOR = 100000; 
	//public static  double CONVERGENCE_NO_CHANGE_EPSILON = 0.001;
	public static  int INIT_MAX_EPOCHS = 100000;

	public static  int INIT_RUNS = 100; 
	public static  int INIT_CANDIDATES_COUNT = 1;
	public static boolean INIT_SHUFFLE_IS = false;
	public static boolean INIT_BATCH_IS = false;

	public static boolean HIDDEN_REPRESENTATION_IS = true;
	public static int HIDDEN_REPRESENTATION_EACH = 1; 
	public static int HIDDEN_REPRESENTATION_AFTER = 200;
	public static int HIDDEN_REPRESENTATION_ONLY_EACH = 50;

	public static  boolean PRINT_NETWORK_IS = true; 
	
	public static double INIT_NORMAL_DISTRIBUTION_SIGMA = 1.25; 
	public static double INIT_LAMBDA = 0.03; 
	public static double INIT_MOMENTUM = 0.1; 
	public static boolean INIT_MOMENTUM_IS = false; 

	public static  double TRY_NORMAL_DISTRIBUTION_SIGMA[] = {2.3}; 
	//public static  double TRY_NORMAL_DISTRIBUTION_SIGMA[] = {1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2}; 
	public static  double TRY_LAMBDA[] = {0.7}; 
	//public static  double TRY_LAMBDA[] = {0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2}; 
	//public static  double TRY_LAMBDA[] = {0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5}; 
	//public static  double TRY_NOISE_SPAN[] = {0.0, 0.003, 0.01, 0.03, 0.1, 0.3}; 
	//public static  double TRY_MULTIPLY_WEIGHTS[] = {1.0, 1.00001, 1.00003, 1.0001, 1.0003, 1.001}; 
	
	public static  double TRY_MOMENTUM[] = {0.0};
	//public static  double TRY_MOMENTUM[] = {-1, -0.3, -0.1, -0.03, -0.01, -0.003, -0.001, 0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}; 

	public static  double INIT_NORMAL_DISTRIBUTION_MU = 0;
	public static  double NORMAL_DISTRIBUTION_SPAN = 15; 

	public static  String[] MEASURE_HEADINGS = {"epoch", "err", "sigma", "lambda", "momentum", "h_dist","h_f_b_dist","m_avg_w","m_sim", "first_second", "o_f_b_dist", "in_triangle"}; 

	public static Map<Integer, String> MEASURE_RUN_ID = new HashMap<Integer, String>(); 
	
	//TODO activation on hidden networks 
	//epoch
	public static  int MEASURE_EPOCH = 0; 

	//error function (RMSE) 
	public static  int MEASURE_ERROR = 1;

	public static  int MEASURE_SIGMA = 2; 
	public static  int MEASURE_LAMBDA = 3; 

	public static  int MEASURE_MOMENTUM = 4;

	//avg of dist(h_i - h_j) i \neq j where h_i is a hidden activation for input i
	//intuitively: internal representation difference 
	public static  int MEASURE_HIDDEN_DIST = 5;

	//avg distance between forward and backward activations on hidden layer
	public static  int MEASURE_HIDDEN_FOR_BACK_DIST = 6;

	//avg weight of matrixes 
	public static  int MEASURE_MATRIX_AVG_W = 7;

	//sum of |a_{ij} - b_{ij}| per pairs (HO, HI) and (OH, IH) 
	public static  int MEASURE_MATRIX_SIMILARITY = 8;

	//ratio of (a_1, a_2) where a_i is the i-th biggest output 
	public static  int MEASURE_FIRST_SECOND_RATIO = 9;

	//public static  int MEASURE_NOISE_SPAN = 9; 
	//public static  int MEASURE_MULTIPLY_WEIGHTS = 9; 

	//avg distance between forward and backward activations on their output layers (forward layer 2, backward layer 0) 
	public static  int MEASURE_OUTPUT_FOR_BACK_DIST = 10;

	//check if some point is inside a polygon from others 
	public static  int MEASURE_IN_TRIANGLE = 11;

	public static  int MEASURE_COUNT = 12;  

	//public static  int[] MEASURE_GROUP_BY_COLS = {MEASURE_ERROR, MEASURE_SIGMA, MEASURE_LAMBDA, MEASURE_IN_TRIANGLE};
	public static  int[] MEASURE_GROUP_BY_COLS = {MEASURE_ERROR, MEASURE_SIGMA, MEASURE_LAMBDA};
	
	public static  int MEASURE_GROUP_BY = MEASURE_ERROR;  

	public static Random random = new Random(); 
	//public static double INIT_NOISE_SPAN = 0.00; 
	//public static double INIT_MULTIPLY_WEIGHTS = 1.001; 

	private ArrayList<Double>[] measures = null; 

	public static ArrayList<double[]> pre_measure = null;
	public static ArrayList<double[]> post_measure = null; 

	public static ArrayList<ArrayList<RealVector[]>> hidden_repre_all = null;
	public static ArrayList<RealVector[]> hidden_repre_cur = null; 

	private RealMatrix IH; 
	private RealMatrix HO;  
	private RealMatrix OH; 
	private RealMatrix HI;

	//Momentum matrices 
	private double[][] MOM_IH; 
	private double[][] MOM_HO;  
	private double[][] MOM_OH; 
	private double[][] MOM_HI;
	
	//Batch matrices 
	private double[][] BATCH_IH; 
	private double[][] BATCH_HO;  
	private double[][] BATCH_OH; 
	private double[][] BATCH_HI;

	private static String RUN_ID = null;  
	
	public static double run(BAL override_network) throws FileNotFoundException{
		BAL.INIT_NORMAL_DISTRIBUTION_SIGMA = BAL.TRY_NORMAL_DISTRIBUTION_SIGMA[random.nextInt(BAL.TRY_NORMAL_DISTRIBUTION_SIGMA.length)]; 
		BAL.INIT_LAMBDA = BAL.TRY_LAMBDA[random.nextInt(BAL.TRY_LAMBDA.length)];
		BAL.INIT_MOMENTUM = BAL.TRY_MOMENTUM[random.nextInt(BAL.TRY_MOMENTUM.length)];
		//BAL.INIT_NOISE_SPAN = BAL.TRY_NOISE_SPAN[random.nextInt(BAL.TRY_NOISE_SPAN.length)];
		//BAL.INIT_MULTIPLY_WEIGHTS = BAL.TRY_MULTIPLY_WEIGHTS[random.nextInt(BAL.TRY_MULTIPLY_WEIGHTS.length)];

		int h_size = BAL.INIT_HIDDEN_LAYER_SIZE; 
		double lambda = BAL.INIT_LAMBDA; 
		int max_epoch = BAL.INIT_MAX_EPOCHS; 

		generateRunId(); 
		
		RealMatrix inputs = BAL.loadFromFile(BAL.INPUT_FILEPATH);
		RealMatrix outputs = BAL.loadFromFile(BAL.OUTPUT_FILEPATH);

		if(HIDDEN_REPRESENTATION_IS){
			hidden_repre_cur = new ArrayList<RealVector[]>();
		}

		//select the "best" candidate network 
		double mav=0.0; 
		double in_points_best = 1000.0; 
		BAL network = new BAL(inputs.getColumnDimension(), h_size, outputs.getColumnDimension()); 
		for(int i=0; i<BAL.INIT_CANDIDATES_COUNT; i++){
			BAL N = new BAL(inputs.getColumnDimension(), h_size, outputs.getColumnDimension()); 
			double[] measure = N.measure(0, inputs, outputs); 
			double hd =  measure[BAL.MEASURE_HIDDEN_DIST];
			double in_points = measure[BAL.MEASURE_IN_TRIANGLE];
			
			if(in_points < in_points_best){
				in_points_best = in_points;
				mav = hd;
				network = N;
			}
			if(in_points == in_points_best && hd > mav){
				mav = hd;
				network = N; 
			}
		}
		
		if(override_network != null){
			network = override_network; 
		}

		//BAL.state_on_begin = ""; 
		//BAL.state_on_begin = matrixToRowString(network.IH) + matrixToRowString(network.HO) + matrixToRowString(network.OH) + matrixToRowString(network.HI); 
	
		
		if(PRINT_NETWORK_IS){
			log.println("----------Network before run: --------------"); 
			log.println(network.printNetwork());
			
			PrintWriter pw = new PrintWriter("data/networks/" + RUN_ID + "_pre.bal"); 
			pw.write(network.printNetwork());
			pw.close(); 
		}

		if(MEASURE_IS) { 
			MEASURE_RUN_ID.put(pre_measure.size(), RUN_ID);
			pre_measure.add(network.measure(0, inputs, outputs));
		}

		//order for shuffling 
		ArrayList<Integer> order = new ArrayList<Integer>(inputs.getRowDimension());
		for(int i=0; i<inputs.getRowDimension() ; i++){
			order.add(i); 
		}

		//HISTORY, INPUT_ID, VECTOR_ENTRY
		RealVector last_outputs[] = new RealVector[inputs.getRowDimension()];
		for(int i=0; i<inputs.getRowDimension() ; i++){
			last_outputs[i] = inputs.getRowVector(i); // fill it with "random" 
		}
		int no_change_epochs=0; 

		//Main learning loop 
		int epochs=0;
		for(epochs=0; epochs<max_epoch ; epochs++){
			if(MEASURE_IS && (MEASURE_SAVE_AFTER_EACH_RUN && epochs % BAL.MEASURE_RECORD_EACH == 0)){
				//log.println(network.evaluate(inputs, outputs));
				network.measure(epochs, inputs, outputs);
			}

			// which hidden representations should be saved 
			RealVector[] hr = null; 
			boolean is_hr = HIDDEN_REPRESENTATION_IS && ((epochs < HIDDEN_REPRESENTATION_AFTER) 
					? epochs % HIDDEN_REPRESENTATION_EACH == 0 
					: epochs % HIDDEN_REPRESENTATION_ONLY_EACH == 0); 
			if(is_hr){
				hr = new RealVector[order.size()]; 
			}

			//TODO Consider as a MEASURE (avg_weight_change) 
			double avg_weight_change = 0.0; 
			if(INIT_SHUFFLE_IS){
				java.util.Collections.shuffle(order);
			}

			boolean is_diffent_output = false; 
			// learn on each given / target mapping 
			for(int order_i : order){
				RealVector in = inputs.getRowVector(order_i);
				RealVector out = outputs.getRowVector(order_i);

				avg_weight_change += network.learn(in, out, lambda);

				RealVector[] forwardPass = network.forwardPass(in); 
				
				if(is_hr){
					hr[order_i] = forwardPass[1]; 
				}
				
				//check if change
				BAL.postprocessOutput(forwardPass[2]);
				is_diffent_output = is_diffent_output || (last_outputs[order_i].getDistance(forwardPass[2]) > DOUBLE_EPSILON);  
				last_outputs[order_i] = forwardPass[2];
				 
			}
			
			/*
			System.out.println("Last outputs, no_change="+no_change_epochs+ " is_different_output="+is_diffent_output+": ");
			for(int i=0; i<last_outputs.length ; i++){
				System.out.print("  " + i + ":" + printVector(last_outputs[i]));
			}*/
			

			if(is_hr){
				hidden_repre_cur.add(hr); 
			}

			// no output change for CONVERGENCE_NO_CHANGE_FOR
			no_change_epochs = (is_diffent_output) ? 0 : no_change_epochs + 1; 
			if(no_change_epochs >= CONVERGENCE_NO_CHANGE_FOR){
				log.println("Training stopped at epoch=" + epochs + " as no output change occured in last " + CONVERGENCE_NO_CHANGE_FOR + "epochs");
				break;
			}
		}

		double network_result = network.evaluate(inputs, outputs);

		if(PRINT_NETWORK_IS){
			log.println("---------- Network after run: --------------");
			log.println(network.printNetwork());
			
			PrintWriter pw = new PrintWriter("data/networks/" + RUN_ID + "_post.bal"); 
			pw.write(network.printNetwork());
			pw.close(); 
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

		if(MEASURE_IS) {
			post_measure.add(network.measure(epochs, inputs, outputs));
		}

		if(HIDDEN_REPRESENTATION_IS){
			hidden_repre_all.add(hidden_repre_cur); 
		}

		//log.println(network.printNetwork());

		if(BAL.MEASURE_IS && BAL.MEASURE_SAVE_AFTER_EACH_RUN){
			network.saveMeasures("data/" + RUN_ID + "_measure_" + ((int)network.evaluate(inputs, outputs)) + "_" + ".dat");
		}

		log.println("Epochs=" + epochs);
		log.println("Result=" + network_result);
		System.out.println("Epochs=" + epochs);
		System.out.println("Result=" + network_result);
		
		return network_result; 
	}

	//interpret activations on the output layer 
	//for example map continuous [0,1] data to discrete {0, 1} 
	public static void postprocessOutput(RealVector out){

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

		return MatrixUtils.createRealMatrix(matrix_data); 
	}
	
	public static String matrixToRowString(RealMatrix m){
		String result = "";
		for(int i=0; i<m.getRowDimension() ; i++){
			for(int j=0; j<m.getColumnDimension() ; j++){
				result += " " + ((Double)m.getEntry(i, j)).toString(); 
			}
		}
		return result; 
	}

	//Creates a BAL network with layer sizes [in_size, h_size, out_size] 
	public BAL(int in_size, int h_size, int out_size) {
		log.println("Creating BAL of size ["+in_size + ","+h_size + ","+out_size + "] RunId=" + RUN_ID);
		//+1 stands for biases 
		//we use matrix premultiply and vertical vectors A*v 
		this.IH = createInitMatrix(in_size+1, h_size);
		this.HO = createInitMatrix(h_size+1, out_size);
		this.OH = createInitMatrix(out_size+1, h_size);
		this.HI = createInitMatrix(h_size+1, in_size);
		
		//#DEVELOPER for special 4-2-4 
		/*
		for(int i=0; i<IH.getRowDimension() ; i++){
			for(int j=0; j<IH.getColumnDimension() ; j++){
				this.OH.setEntry(i, j, this.IH.getEntry(i, j)); 
			}
		} 
		for(int i=0; i<HO.getRowDimension() ; i++){
			for(int j=0; j<HO.getColumnDimension() ; j++){
				this.HI.setEntry(i, j, this.HO.getEntry(i, j)); 
			}
		}*/
		
		this.BAL_construct_other(); 
		 
	}
	
	private void BAL_construct_other(){
			
		int in_size = this.HI.getColumnDimension();
		int h_size = this.IH.getColumnDimension();
		int out_size = this.HO.getColumnDimension(); 
		
		if(INIT_MOMENTUM_IS){
			this.MOM_IH = new double[in_size+1][h_size];
			this.MOM_HO = new double[h_size+1][out_size];
			this.MOM_OH = new double[out_size+1][h_size];
			this.MOM_HI = new double[h_size+1][in_size];
		}
		if(INIT_BATCH_IS){
			this.BATCH_IH = new double[in_size+1][h_size];
			this.BATCH_HO = new double[h_size+1][out_size];
			this.BATCH_OH = new double[out_size+1][h_size];
			this.BATCH_HI = new double[h_size+1][in_size];
		}

		this.measures = new ArrayList[MEASURE_COUNT]; 
		for(int i=0; i<MEASURE_COUNT; i++){
			this.measures[i] = new ArrayList<Double>();
		}
	}
	
	public static RealMatrix loadMatrixFromReader(BufferedReader reader) throws IOException{
		String[] tokens = reader.readLine().split(" ");
		int rows = Integer.parseInt(tokens[0]);  
		int cols = Integer.parseInt(tokens[1]); 
		
		System.out.println("Loading matrix [" + cols + "," + rows + "]");
		
		double[][] real_matrix = new double[rows][cols]; 
		
		for(int i=0; i<rows; i++){
			tokens = reader.readLine().split(" ");
			
			for(int j=0; j<cols; j++){
				real_matrix[i][j] = Double.parseDouble(tokens[j]); 
			}
		}
		
		return MatrixUtils.createRealMatrix(real_matrix); 
	}

	public BAL(String filename) throws IOException{
		log.println("Creating BAL from file '" + filename + "' RunId=" + RUN_ID);
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		
		reader.readLine();
		reader.readLine();
		this.IH = BAL.loadMatrixFromReader(reader);
		reader.readLine();
		this.HO = BAL.loadMatrixFromReader(reader);
		reader.readLine();
		this.OH = BAL.loadMatrixFromReader(reader);
		reader.readLine();
		this.HI = BAL.loadMatrixFromReader(reader);
		
		reader.close(); 
		
		this.BAL_construct_other(); 
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
	private double subLearn(RealMatrix w, RealVector a_pre, RealVector a_post_other, RealVector a_post_self, double lambda, double[][] mom, double[][] batch){
		double avg_change = 0.0; 

		for(int i = 0 ; i < w.getRowDimension() ; i++){
			for(int j = 0 ; j < w.getColumnDimension() ; j++){
				double w_value = w.getEntry(i, j); 
				double dw = lambda * a_pre.getEntry(i) * (a_post_other.getEntry(j) - a_post_self.getEntry(j));

				if(INIT_BATCH_IS){
					batch[i][j] += dw; 
				}
				else{
					w.setEntry(i, j, w_value + dw + (INIT_MOMENTUM_IS ? BAL.INIT_MOMENTUM * mom[i][j] : 0.0));
				}

				if(INIT_MOMENTUM_IS){
					mom[i][j] = dw; 
				}
				
				avg_change += Math.abs(dw / w_value);
			}
		}

		return avg_change / ((double)(w.getRowDimension() * w.getColumnDimension()));
	}
	


	//learns on a weight matrix, other parameters are activations on needed layers 
	//\delta w_{pq}^F = \lambda a_p^{F}(a_q^{B} - a_q^{F})
	//\delta w_ij = lamda * a_pre * (a_post_other - a_post_self)  
	private double subCHLLearn(RealMatrix w, RealVector a_plus_i, RealVector a_plus_j, RealVector a_minus_j, RealVector a_minus_i, double lambda){
		double avg_change = 0.0; 

		for(int i = 0 ; i < w.getRowDimension() ; i++){
			for(int j = 0 ; j < w.getColumnDimension() ; j++){
				//System.out.println("  " + i + "," + j);
				double w_value = w.getEntry(i, j);
				
				double dw = 0.0;
				if(a_plus_i.getDimension() <= i || a_plus_j.getDimension() <= j || a_minus_j.getDimension() <= j || a_minus_i.getDimension() <= i){
					dw = lambda * a_plus_i.getEntry(i) * (a_minus_j.getEntry(j) - a_plus_j.getEntry(j));
				}
				else{
					dw = lambda * ((a_plus_i.getEntry(i) * a_plus_j.getEntry(j)) - (a_minus_i.getEntry(i) * a_minus_j.getEntry(j)));
				}
				
				w.setEntry(i, j, w_value + dw);
				avg_change += Math.abs(dw / w_value);
			}
		}

		return avg_change / ((double)(w.getRowDimension() * w.getColumnDimension()));
	}

	private void resetTwoDimArray(double[][] arr){
		for(int i=0; i<arr.length ; i++){
			for(int j=0; j<arr[i].length ; j++){
				arr[i][j] = 0.0; 
			}
		}
	}

	private void updateTwoDimArray(RealMatrix m, double[][] arr){
		for(int i=0; i<arr.length ; i++){
			for(int j=0; j<arr[i].length ; j++){
				m.setEntry(i, j, m.getEntry(i, j) + arr[i][j]); 
			}
		}
	}
	
	//learn on one input-output mapping
	// returns avg weight change 
	public double learn(RealVector in, RealVector target, double lambda){
		//forward and backward activation
		RealVector[] forward = this.forwardPass(in);
		RealVector[] backward = this.backwardPass(target);
		double avg_change_ih = 0.0; 
		double avg_change_oh = 0.0; 

		if(INIT_BATCH_IS){
			resetTwoDimArray(this.BATCH_IH);
			resetTwoDimArray(this.BATCH_HO);
			resetTwoDimArray(this.BATCH_OH);
			resetTwoDimArray(this.BATCH_HI);
		}
		
		/*
		System.out.println("IH dim: " + this.IH.getRowDimension() + "," + this.IH.getColumnDimension());
		System.out.println("HO dim: " + this.HO.getRowDimension() + "," + this.HO.getColumnDimension());
		System.out.println("OH dim: " + this.OH.getRowDimension() + "," + this.OH.getColumnDimension());
		System.out.println("HI dim: " + this.HI.getRowDimension() + "," + this.HI.getColumnDimension());
		for(int i=0; i<3; i++){
			System.out.println("Forward " + i + ": " + forward[i].getDimension());
		}
		for(int i=0; i<3; i++){
			System.out.println("Backward " + i + ": " + backward[i].getDimension());
		}*/
		
		//learn
		if(SYMMETRIC_WEIGHT_UPDATE){
			avg_change_ih += subCHLLearn(this.IH, forward[0], forward[1], backward[1], backward[0], lambda);
			avg_change_ih += subCHLLearn(this.HO, forward[1], forward[2], backward[2], backward[1], lambda);
			avg_change_ih += subCHLLearn(this.OH, backward[2], backward[1], forward[1], forward[2], lambda);
			avg_change_ih += subCHLLearn(this.HI, backward[1], backward[0], forward[0], forward[1], lambda);
		}
		else{
			avg_change_ih += subLearn(this.IH, forward[0], backward[1], forward[1], lambda, this.MOM_IH, this.BATCH_IH); 
			avg_change_oh += subLearn(this.HO, forward[1], backward[2], forward[2], lambda, this.MOM_HO, this.BATCH_HO); 
			avg_change_ih += subLearn(this.OH, backward[2], forward[1], backward[1], lambda, this.MOM_OH, this.BATCH_OH); 
			avg_change_oh += subLearn(this.HI, backward[1], forward[0], backward[0], lambda, this.MOM_HI, this.BATCH_HI);
		}

		if(INIT_BATCH_IS){
			updateTwoDimArray(this.IH, this.BATCH_IH);
			updateTwoDimArray(this.HO, this.BATCH_HO);
			updateTwoDimArray(this.OH, this.BATCH_OH);
			updateTwoDimArray(this.HI, this.BATCH_HI);
		}
		
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
		BAL.postprocessOutput(result);
		
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

	/** Fetches angle relative to screen centre point
	 * where 3 O'Clock is 0 and 12 O'Clock is 270 degrees
	 * 
	 * @param screenPoint
	 * @return angle in degress from 0-360.
	 */
	public static double getAngle(double dx, double dy)
	{
	    double inRads = Math.atan2(dy,dx);

	    // We need to map to coord system when 0 degree is at 3 O'clock, 270 at 12 O'clock
	    if (inRads < 0)
	        inRads = Math.abs(inRads);
	    else
	        inRads = 2*Math.PI - inRads;

	    return Math.toDegrees(inRads);
	}
	
	//collect monitoring data, epoch is used as identifier
	//  !this data is also stored into measures array 
	public double[] measure(int epoch, RealMatrix in, RealMatrix target){
		double n = in.getRowDimension(); 

		if(MEASURE_EPOCH < MEASURE_COUNT) this.measures[MEASURE_EPOCH].add((double)epoch); 
		if(MEASURE_SIGMA < MEASURE_COUNT) this.measures[MEASURE_SIGMA].add(BAL.INIT_NORMAL_DISTRIBUTION_SIGMA); 
		if(MEASURE_LAMBDA < MEASURE_COUNT) this.measures[MEASURE_LAMBDA].add(BAL.INIT_LAMBDA); 
		if(MEASURE_MOMENTUM < MEASURE_COUNT) this.measures[MEASURE_MOMENTUM].add(BAL.INIT_MOMENTUM); 
		//this.measures[MEASURE_NOISE_SPAN].add(BAL.INIT_NOISE_SPAN); 
		//this.measures[MEASURE_MULTIPLY_WEIGHTS].add(BAL.INIT_MULTIPLY_WEIGHTS - 1); 

		if(MEASURE_ERROR < MEASURE_COUNT) this.measures[MEASURE_ERROR].add(this.evaluate(in, target)); 

		if(MEASURE_HIDDEN_FOR_BACK_DIST < MEASURE_COUNT 
				|| MEASURE_OUTPUT_FOR_BACK_DIST < MEASURE_COUNT 
				|| MEASURE_FIRST_SECOND_RATIO < MEASURE_COUNT 
				|| MEASURE_HIDDEN_DIST < MEASURE_COUNT ){
			ArrayList<RealVector> forward_hiddens = new ArrayList<RealVector>(); 
			double hidden_dist = 0.0;
			double hidden_for_back_dist = 0.0;
			double output_for_back_dist = 0.0;

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
			if(MEASURE_HIDDEN_FOR_BACK_DIST < MEASURE_COUNT) this.measures[MEASURE_HIDDEN_FOR_BACK_DIST].add(hidden_for_back_dist); 
			if(MEASURE_OUTPUT_FOR_BACK_DIST < MEASURE_COUNT) this.measures[MEASURE_OUTPUT_FOR_BACK_DIST].add(output_for_back_dist); 
			if(MEASURE_FIRST_SECOND_RATIO < MEASURE_COUNT) this.measures[MEASURE_FIRST_SECOND_RATIO].add(first_second_sum); 

			if(MEASURE_HIDDEN_DIST < MEASURE_COUNT){
				for(int i=0; i<forward_hiddens.size() ; i++){
					for(int j=i+1; j<forward_hiddens.size() ; j++){
						hidden_dist += forward_hiddens.get(i).getDistance(forward_hiddens.get(j)) / (forward_hiddens.size() * (forward_hiddens.size() + 1) / 2); 
					}
				}

				this.measures[MEASURE_HIDDEN_DIST].add(hidden_dist);  
			}
			
			if(MEASURE_IN_TRIANGLE < MEASURE_COUNT){
				ArrayList<Point> hidden_points = new ArrayList<Point>(); 
				for(int i=0; i<forward_hiddens.size() ; i++){
					hidden_points.add(new Point((int)(1000.0 * forward_hiddens.get(i).getEntry(0)), (int)(1000.0 * forward_hiddens.get(i).getEntry(1)))); 
				}
				ArrayList<Point> convex_hull = ConvexHull.execute(hidden_points); 
				this.measures[MEASURE_IN_TRIANGLE].add((double)(hidden_points.size() - convex_hull.size()));
				
				log.println("Hidden points");
				log.println(hidden_points);
				log.println("ConvexHull points");
				log.println(convex_hull);
				log.println("End");
			}
		}

		if(MEASURE_MATRIX_AVG_W < MEASURE_COUNT){
			double matrix_avg_w = 0.0;
			matrix_avg_w = (sumAbsoluteValuesOfMatrixEntries(this.IH) + sumAbsoluteValuesOfMatrixEntries(this.HO) + sumAbsoluteValuesOfMatrixEntries(this.OH) + sumAbsoluteValuesOfMatrixEntries(this.IH)) / (this.IH.getColumnDimension()*this.IH.getRowDimension() + this.HO.getColumnDimension()*this.HO.getRowDimension()+ this.OH.getColumnDimension()*this.OH.getRowDimension()+ this.HI.getColumnDimension()*this.HI.getRowDimension()); 
			this.measures[MEASURE_MATRIX_AVG_W].add(matrix_avg_w);
		}

		if(MEASURE_MATRIX_SIMILARITY < MEASURE_COUNT){
			double matrix_similarity = 0.0;
			if(MEASURE_MATRIX_SIMILARITY >= 0 && this.HO.getColumnDimension() == this.HI.getColumnDimension() && this.HO.getRowDimension() == this.HI.getRowDimension()){
				RealMatrix diff_HO_HI = this.HO.subtract(this.HI); 
				RealMatrix diff_OH_IH = this.OH.subtract(this.IH);
				matrix_similarity = (sumAbsoluteValuesOfMatrixEntries(diff_HO_HI) + sumAbsoluteValuesOfMatrixEntries(diff_OH_IH)) / (this.IH.getColumnDimension()*this.IH.getRowDimension() + this.HI.getColumnDimension()*this.HI.getRowDimension());   
				this.measures[MEASURE_MATRIX_SIMILARITY].add(matrix_similarity);
			}
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

		StringBuilder sb = new StringBuilder(); 
		for(int j=0; j<m; j++){
			if(j != 0) sb.append(" ");
			sb.append(BAL.MEASURE_HEADINGS[BAL.MEASURE_GROUP_BY_COLS[j]]);
		}
		sb.append(" success sample_ratio\n");

		List<String> result = new ArrayList<String>(); 
		for(Entry<String, Integer> entry : counts_child.entrySet()){
			Integer child_count = entry.getValue();
			Integer parent_count = counts_parent.get(child2parent.get(entry.getKey())); 
			result.add(entry.getKey() + (100.0*((double)child_count / (double)parent_count)) + " " + child_count + "/" + parent_count);
		}
		Collections.sort(result); 
		for(String s : result){
			sb.append(s + "\n");
		}
		
		log.print(sb.toString());
		System.out.println(sb.toString());
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
				//TODO round 
				sb.append(m.getEntry(i, j));
			}
			sb.append('\n');
		}
		return sb.toString(); 
	}

	public String printNetwork(){
		StringBuilder sb = new StringBuilder(); 
		sb.append((this.IH.getRowDimension()-1) + " " + this.IH.getColumnDimension() + " " + this.HO.getColumnDimension() + "\n");
		sb.append("#IH\n"); 
		sb.append(BAL.printMatrix(this.IH));
		sb.append("#HO\n"); 
		sb.append(BAL.printMatrix(this.HO));
		sb.append("#OH\n"); 
		sb.append(BAL.printMatrix(this.OH));
		sb.append("#HI\n"); 
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
			pre_writer = new PrintWriter("data/" + RUN_ID + "_pre.dat", "UTF-8");
			post_writer = new PrintWriter("data/" + RUN_ID + "_post.dat", "UTF-8");
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
			//#DEVELOPER
			//pre_measure.get(i)[MEASURE_ERROR] = post_measure.get(i)[MEASURE_ERROR];
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

		System.out.println("PreMeasure : GroupBy");
		BAL.measureGroupBY(pre_measure); 
		System.out.println("PostMeasure : GroupBy");
		BAL.measureGroupBY(post_measure); 
		log.print("PreMeasure : Averages");
		BAL.measureAverages(pre_measure); 
		log.print("PostMeasure : Averages");
		BAL.measureAverages(post_measure); 
	}

	public static void printHiddenRepresentations() throws FileNotFoundException, UnsupportedEncodingException{
		if(!HIDDEN_REPRESENTATION_IS){
			return;
		}

		for(int i=0; i<hidden_repre_all.size() ; i++){
			ArrayList<RealVector[]> priebeh = hidden_repre_all.get(i);

			for(int k=0; k < priebeh.get(0).length ; k++){
				String filename = "data/hr/" + ((post_measure.get(i)[MEASURE_ERROR] == 0.0) ? "good" : "bad") + "/" + MEASURE_RUN_ID.get(i) + "_" + k + ".dat";
				PrintWriter hr_writer = new PrintWriter(filename, "UTF-8");
				for(RealVector[] vectors : priebeh){
					RealVector v = vectors[k]; 
					for(int a =0 ; a < v.getDimension() - 1 ; a++ ){ // -1 stands for bias 
						if(a != 0) hr_writer.print(' ');
						hr_writer.print(v.getEntry(a)); 
					}
					hr_writer.println(); 
				}
				hr_writer.close(); 
			}
		}
	}

	public static void generateRunId(){
		RUN_ID = INPUT_FILEPATH.substring(0, INPUT_FILEPATH.indexOf('.')) + "_" + (System.currentTimeMillis()) + "_" + INIT_HIDDEN_LAYER_SIZE;
	}

	//manage IO and run BAL 
	public static void main(String[] args) throws IOException {
		/*
		for(int h=5; h<=144; h += h/8 + 1){
			initMultidimensional("k12", h);
			experiment();
		} 
		 */ 
		experiment(); 
	}
	
	public static void experiment() throws IOException{
		generateRunId(); 
		
		pre_measure = new ArrayList<double[]>();
		post_measure = new ArrayList<double[]>();
		if(HIDDEN_REPRESENTATION_IS){
			hidden_repre_all = new ArrayList<ArrayList<RealVector[]>>(); 
		}

		log = new PrintWriter("data/" + RUN_ID + ".log");
		PrintWriter file_initial_state = new PrintWriter("data/" + RUN_ID + ".train");
		file_initial_state.println("error IH HO OH HI");
		
		/*
		File folder = new File("data/hr/good/");
		Set<String> filenames = new HashSet<String>(); 
		int file_c=0; 
		for(File file : folder.listFiles()){
			String filename = file.getName();  
			if(!filename.endsWith(".png")){
				continue; 
			}

			System.out.println("=== Loading " + filename + " (" + file_c + "/" + filenames.size() + ")");
			file_c++; 
			
			System.out.println(filename);
			int pos = filename.lastIndexOf('_'); 
			if(pos > 0){
				filenames.add(filename.substring(0, pos));
			}
		}
		
		for(String filename : filenames){
			String filepath = "data/networks/" + filename + "_pre.bal";
			if(!new File(filepath).exists()){
				continue; 
			}
			BAL network = new BAL(filepath); */ 
			
			for(int i=0 ; i<BAL.INIT_RUNS ; i++){
				long start_time = System.currentTimeMillis(); 
	
				log.println("  ======== " + i + "/" + BAL.INIT_RUNS + " ==============");
				System.out.println("  ======== " + i + "/" + BAL.INIT_RUNS + " ==============");
	
				Double error = BAL.run(null); 
	
				long run_time = (System.currentTimeMillis() - start_time); 
				log.println("  RunTime=" + run_time);
				System.out.println("  RunTime=" + run_time);
				//file_initial_state.println(error.toString() + BAL.state_on_begin); //TODO 
			}
		//}

		printPreAndPostMeasures();
		printHiddenRepresentations(); 
		log.close(); 
	}

	public static void initMultidimensional(String input_prefix, int hidden_size){
		MEASURE_IS = true; 
		MEASURE_SAVE_AFTER_EACH_RUN = false; 
		MEASURE_RECORD_EACH = 50;

		INPUT_FILEPATH = input_prefix + ".in"; 
		OUTPUT_FILEPATH = input_prefix + ".out"; 
		INIT_HIDDEN_LAYER_SIZE = hidden_size; 

		CONVERGENCE_WEIGHT_EPSILON = 0.0; 

		CONVERGENCE_NO_CHANGE_FOR = 10; 
		INIT_MAX_EPOCHS = 1000;

		INIT_RUNS = 1; 
		INIT_CANDIDATES_COUNT = 0;

		PRINT_NETWORK_IS = false; 

		TRY_NORMAL_DISTRIBUTION_SIGMA = new double[] {1.0}; 
		TRY_LAMBDA = new double[] {0.2}; 
		TRY_MOMENTUM = new double[] {0.0}; 
	}
}
