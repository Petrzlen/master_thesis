//TODO struktura, uvod 
//TODO symmetric version of BAL, energy   
//TODO generec implementation
//TODO almeida-pineda iterative activation
//TODO dynamicka rychlost ucenia 
  // MEASURE activation change 
  // kopirovat priebeh 
  // momemntum 

//TODO patternSuccess, bitSuccess  
//TODO GeneRec Obojstranne vysokorozmerne

//TODO GeneRec na nase autoasociativne problemy 
//  TODO iterative activation calculation 

//TODO Refactor 
//TODO some memory leak / time explosion when: 
//	public static  int INIT_MAX_EPOCHS = 1000000;
//	public static  int INIT_RUNS = 250; 	
//	public static  int CONVERGENCE_NO_CHANGE_FOR = ; 
//	public static  int CONVERGENCE_NO_CHANGE_FOR = 10000000; 

//TODO O'Really - kedy podobny backpropagation 
//  -- ci sa naozaj snazi minimalizovat gradient 

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
//TODO iny rozmer tasku (2-1-2), (8-3-8), (16-4-16)
//TODO kvazi momentum -> odtlacanie hidden reprezentacii, -\delta w(t-1) 
//TODO pocet epoch potrebnych na konvergenciu
//TODO nie autoassoc ale permutovat vystupy (napr. 1000 na 0100)
//TODO reprezentacia SUC/ERR 
//TODO m_sim, h_f_b_dist -> jeden nasobkom druheho bias
//TODO spektralna analyza
//TODO bipolarna [-1, 1] vstupy
//h_size = 3 => [9,10,1] for errors [0.0, 1.0, 2.0] 

//TODO run simulations with adding noise / multiplying weights

//TODO speed up by only calculating what is necessary 

import java.awt.Point;
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

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class BAL {
	//manage IO and run BAL 
	public static void main(String[] args) throws IOException {
		//experiment_Default();
		//experiment_DifferentHiddenSizes("k12");
		//experiment_RerunGoodBad();
		experiment_TestImplementation();
	}
	
	private static boolean IS_PRINT = false; 

	private static int BAL_WEIGHT_UPDATE = 1; 
	private static int CHL_WEIGHT_UPDATE = 2; //TODO must to be iterative activation
	private static int BAL_RECIRC_WEIGHT_UPDATE = 3; 
	private static int GENEREC_WEIGHT_UPDATE = 4; // => INIT_SYMMETRIC_IS = true 
	private static int WEIGHT_UPDATE_TYPE = BAL_WEIGHT_UPDATE;
	private static boolean INIT_RECIRCULATION_IS = (WEIGHT_UPDATE_TYPE == CHL_WEIGHT_UPDATE || WEIGHT_UPDATE_TYPE == BAL_RECIRC_WEIGHT_UPDATE || WEIGHT_UPDATE_TYPE == GENEREC_WEIGHT_UPDATE); 
	private static double RECIRCULATION_EPSILON = 0.01; //if the max unit activation change is less the RECIRCULATION_EPSILON, it will stop 
	private static int RECIRCULATION_ITERATIONS_MAX = 20; //maximum number of iterations to approximate the underlying dynamic system  
	private static boolean RECIRCULATION_USE_AVERAGE_WHEN_OSCILATING = false; // average of last two activations will be used instead of the last one (intuition: more stable) 
	
	private static PrintWriter log = null; 

	public static double DOUBLE_EPSILON = 0.001;

	public static  boolean MEASURE_IS = true; 
	public static boolean MEASURE_SAVE_AFTER_EACH_RUN = true; 
	public static  int MEASURE_RECORD_EACH = 100;

	public static  String INPUT_FILEPATH = "auto4.in"; 
	public static  String OUTPUT_FILEPATH = "auto4.in"; 
	public static  int INIT_HIDDEN_LAYER_SIZE = 2 ; 

	public static  int INIT_MAX_EPOCHS = 500000;
	public static  int INIT_RUNS = 100; 
	public static  int INIT_CANDIDATES_COUNT = 1000;
	public static boolean INIT_SHUFFLE_IS = false;
	public static boolean INIT_BATCH_IS = false;
	public static boolean INIT_SYMMETRIC_IS = true; 

	//which hidden layer neurons are active (bias is not counted), used for dropout
	public static boolean DROPOUT_IS = false; //TODO check some runs, it gives error 
	private boolean[] active_hidden; 
	private static boolean[] all_true_active_hidden; // a mock which says "all hidden units are active" 

	public static  int CONVERGENCE_NO_CHANGE_FOR = 500000; 
	public static boolean STOP_IF_NO_ERROR = true; 

	public static double INIT_NORMAL_DISTRIBUTION_SIGMA = 2.3; 
	//public static double TRY_NORMAL_DISTRIBUTION_SIGMA[] = {1.0}; // generec  
	public static double TRY_NORMAL_DISTRIBUTION_SIGMA[] = {2.3}; 
	//public static  double TRY_NORMAL_DISTRIBUTION_SIGMA[] = {1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2};
	//public static  double TRY_NORMAL_DISTRIBUTION_SIGMA[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}; 
	//public static double TRY_NORMAL_DISTRIBUTION_SIGMA[] = {0.3, 0.5, 0.7, 1.0};
	
	public static double INIT_LAMBDA = 0.7; 
	//public static  double TRY_LAMBDA[] = {0.7}; 
	//public static  double TRY_LAMBDA[] = {0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2}; 
	public static  double TRY_LAMBDA[] = {0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2};
	//public static double TRY_LAMBDA[] = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2}; 

	//public static  double TRY_NOISE_SPAN[] = {0.0, 0.003, 0.01, 0.03, 0.1, 0.3}; 
	//public static  double TRY_MULTIPLY_WEIGHTS[] = {1.0, 1.00001, 1.00003, 1.0001, 1.0003, 1.001}; 

	public static boolean INIT_MOMENTUM_IS = false;  // a performance flag 
	public static double INIT_MOMENTUM = 0.1;  
	public static  double TRY_MOMENTUM[] = {0.1};
	//public static  double TRY_MOMENTUM[] = {-1, -0.3, -0.1, -0.03, -0.01, -0.003, -0.001, 0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}; 

	public static  double INIT_NORMAL_DISTRIBUTION_MU = 0;
	public static  double NORMAL_DISTRIBUTION_SPAN = 15; 

	// save the hidden representations 
	public static boolean HIDDEN_REPRESENTATION_IS = false;
	public static String HIDDEN_REPRESENTATION_DIRECTORY = "data/hr/"; 
	public static int HIDDEN_REPRESENTATION_EACH = 1; 
	public static int HIDDEN_REPRESENTATION_AFTER = 200;
	public static int HIDDEN_REPRESENTATION_ONLY_EACH = 50;

	public static  boolean PRINT_NETWORK_IS = false; //!TODO should be turned-off most of the times ! 

	//=======================  "TELEMETRY" of the network in time =========================== 
	//TODO Consider as a MEASURE (avg_weight_change) 
	public static Map<Integer, String> MEASURE_RUN_ID = new HashMap<Integer, String>(); 
	public static  String[] MEASURE_HEADINGS = {"epoch", "err", "sigma", "lambda", "momentum", "h_dist","h_f_b_dist","m_avg_w","m_sim", "first_second", "o_f_b_dist", "in_triangle", "fluctuation"};
	public static  int MEASURE_EPOCH = 0;
	public static  int MEASURE_ERROR = 1; //error function (RMSE) 
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

	//sum of ratio of (a_1, a_2) where a_i is the i-th biggest output 
	public static  int MEASURE_FIRST_SECOND_RATIO = 9;

	//public static  int MEASURE_NOISE_SPAN = 9; 
	//public static  int MEASURE_MULTIPLY_WEIGHTS = 9; 

	//avg distance between forward and backward activations on their output layers (forward layer 2, backward layer 0) 
	public static  int MEASURE_OUTPUT_FOR_BACK_DIST = 10;

	//check if some point is inside a polygon from others 
	public static  int MEASURE_IN_TRIANGLE = 11;

	//how much differ activations when iterative method is used 
	public static int MEASURE_FLUCTUATION = 12; 
			
	public static int MEASURE_COUNT = 13;  

	//public static  int[] MEASURE_GROUP_BY_COLS = {MEASURE_ERROR, MEASURE_SIGMA, MEASURE_LAMBDA, MEASURE_IN_TRIANGLE};
	public static  int[] MEASURE_GROUP_BY_COLS = {MEASURE_ERROR, MEASURE_SIGMA, MEASURE_LAMBDA, MEASURE_MOMENTUM};

	public static  int MEASURE_GROUP_BY = MEASURE_ERROR;  

	// ================= DATA COLLECTORS ===================
	public static Random random = new Random(); 
	//public static double INIT_NOISE_SPAN = 0.00; 
	//public static double INIT_MULTIPLY_WEIGHTS = 1.001; 

	private ArrayList<Double>[] measures = null; 

	public static ArrayList<double[]> pre_measure = null;
	public static ArrayList<double[]> post_measure = null; 
	public static PrintWriter measure_writer; 

	// TIMELINE of hidden representations 
	public static ArrayList<ArrayList<RealVector[]>> hidden_repre_all = null;
	public static ArrayList<RealVector[]> hidden_repre_cur = null; 
	
	public static ArrayList<Integer> recirc_iter_counts = new ArrayList<Integer>(); 

	// ================= STATE of the network ========================== 
	// .getRowDimension() = with bias
	// .getColumnDimension() = without bias 
	private RealMatrix IH; 
	private RealMatrix HO;  
	private RealMatrix OH; 
	private RealMatrix HI;
	
	private static final int MATRIX_IH = 0; 
	private static final int MATRIX_HO = 1; 
	private static final int MATRIX_OH = 2; 
	private static final int MATRIX_HI = 3; 

	//Momentum matrices 
	private double[][] MOM_IH; 
	private double[][] MOM_HO;  
	private double[][] MOM_OH; 
	private double[][] MOM_HI;

	//Batch matrices (sum all weight changes and update after the current epoch ended) 
	private double[][] BATCH_IH; 
	private double[][] BATCH_HO;  
	private double[][] BATCH_OH; 
	private double[][] BATCH_HI; 

	private static String NETWORK_RUN_ID = null;
	private static int NETWORK_EPOCH = 0; 

	// if override_network != null then the provided netwoek will be used (usually loaded from file) 
	public static double run(BAL override_network) throws FileNotFoundException{
		NETWORK_EPOCH = 0; 
		max_fluctuation = 0.0; 
		if(WEIGHT_UPDATE_TYPE == GENEREC_WEIGHT_UPDATE) INIT_SYMMETRIC_IS = true; 
		
		BAL.INIT_NORMAL_DISTRIBUTION_SIGMA = BAL.TRY_NORMAL_DISTRIBUTION_SIGMA[random.nextInt(BAL.TRY_NORMAL_DISTRIBUTION_SIGMA.length)]; 
		BAL.INIT_LAMBDA = BAL.TRY_LAMBDA[random.nextInt(BAL.TRY_LAMBDA.length)];
		BAL.INIT_MOMENTUM = BAL.TRY_MOMENTUM[random.nextInt(BAL.TRY_MOMENTUM.length)];
		//BAL.INIT_NOISE_SPAN = BAL.TRY_NOISE_SPAN[random.nextInt(BAL.TRY_NOISE_SPAN.length)];
		//BAL.INIT_MULTIPLY_WEIGHTS = BAL.TRY_MULTIPLY_WEIGHTS[random.nextInt(BAL.TRY_MULTIPLY_WEIGHTS.length)];

		int h_size = BAL.INIT_HIDDEN_LAYER_SIZE; 
		double lambda = BAL.INIT_LAMBDA; 
		int max_epoch = BAL.INIT_MAX_EPOCHS; 

		generateNetworkRunId(); 

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

		if(PRINT_NETWORK_IS){
			log.println("----------Network before run: --------------"); 
			log.println(network.printNetwork());

			PrintWriter pw = new PrintWriter("data/networks/" + NETWORK_RUN_ID + "_pre.bal"); 
			pw.write(network.printNetwork());
			pw.close(); 
		}

		if(MEASURE_IS) { 
			MEASURE_RUN_ID.put(pre_measure.size(), NETWORK_RUN_ID);
			pre_measure.add(network.measure(0, inputs, outputs));
		}

		// shuffled order   
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
			NETWORK_EPOCH = epochs; 
			
			if(MEASURE_IS && (MEASURE_SAVE_AFTER_EACH_RUN && epochs % BAL.MEASURE_RECORD_EACH == 0)){
				//log.println(network.evaluate(inputs, outputs));
				network.measure(epochs, inputs, outputs);
			}

			// which hidden representations should be saved 
			RealVector[] hidden_representation = null; 
			boolean is_save_hidden_representation = HIDDEN_REPRESENTATION_IS && ((epochs < HIDDEN_REPRESENTATION_AFTER) 
					? epochs % HIDDEN_REPRESENTATION_EACH == 0 
					: epochs % HIDDEN_REPRESENTATION_ONLY_EACH == 0); 
			if(is_save_hidden_representation){
				hidden_representation = new RealVector[order.size()]; 
			}

			if(INIT_SHUFFLE_IS){
				java.util.Collections.shuffle(order);
			}

			if(INIT_BATCH_IS){
				resetTwoDimArray(network.BATCH_IH);
				resetTwoDimArray(network.BATCH_HO);
				resetTwoDimArray(network.BATCH_OH);
				resetTwoDimArray(network.BATCH_HI);
			}

			// check if different outputs given as the last epoch 
			boolean is_diffent_output = false; 
			
			// learn on each given / target mapping 
			for(int order_i : order){
				RealVector in = inputs.getRowVector(order_i);
				RealVector out = outputs.getRowVector(order_i);

				network.learn(in, out, lambda);

				RealVector[] forwardPass = network.forwardPass(in); 

				if(is_save_hidden_representation){
					hidden_representation[order_i] = forwardPass[1]; 
				}

				//check if change
				BAL.postprocessOutput(forwardPass[2]);
				is_diffent_output = is_diffent_output || (last_outputs[order_i].getDistance(forwardPass[2]) > DOUBLE_EPSILON);  
				last_outputs[order_i] = forwardPass[2];
			}

			if(INIT_BATCH_IS){
				updateTwoDimArray(network.IH, network.BATCH_IH);
				updateTwoDimArray(network.HO, network.BATCH_HO);
				updateTwoDimArray(network.OH, network.BATCH_OH);
				updateTwoDimArray(network.HI, network.BATCH_HI);
			}

			if(is_save_hidden_representation){
				hidden_repre_cur.add(hidden_representation); 
			}

			// no output change for CONVERGENCE_NO_CHANGE_FOR
			no_change_epochs = (is_diffent_output) ? 0 : no_change_epochs + 1; 
			if(no_change_epochs >= CONVERGENCE_NO_CHANGE_FOR){
				log.println("Training stopped at epoch=" + epochs + " as no output change occured in last " + CONVERGENCE_NO_CHANGE_FOR + "epochs");
				break;
			}
			
			// we need to evaluate the performance on each input / output as when non-batch learning the total_error could be changed after weight change 
			if(STOP_IF_NO_ERROR && network.evaluate(inputs, outputs) == 0.0){
				log.println("Training stopped at epoch=" + epochs + " as all outputs given correctly");
				break;
			}
		}

		if(PRINT_NETWORK_IS){
			log.println("---------- Network after run: --------------");
			log.println(network.printNetwork());

			PrintWriter pw = new PrintWriter("data/networks/" + NETWORK_RUN_ID + "_post.bal"); 
			pw.write(network.printNetwork());
			pw.close(); 
		}

		//print forward pass activations 
		for(int i=0 ; i<inputs.getRowDimension(); i++){
			RealVector[] forward = network.forwardPass(inputs.getRowVector(i));

			if(PRINT_NETWORK_IS){
				log.println("Forward pass:");
				for(int j=0; j<forward.length; j++){
					log.print(BAL.printVector(forward[j]));
				}
			}

			BAL.postprocessOutput(forward[2]);
			log.print("Given:   " + BAL.printVector(forward[2]));
			log.print("Expected:" + BAL.printVector(outputs.getRowVector(i)));
			log.println();
		}
		//print backward pass activations  
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

		if(BAL.MEASURE_IS && BAL.MEASURE_SAVE_AFTER_EACH_RUN){
			network.saveMeasures(NETWORK_RUN_ID, measure_writer);
		}

		// Print out basics 
		double network_result = network.evaluate(inputs, outputs);

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

	public static void makeSymmetric(RealMatrix target, RealMatrix from, int rows, int cols) { 
		for(int i=0 ; i<rows ; i++){
			for(int j=0 ; j < cols ; j++){
				target.setEntry(i, j, from.getEntry(j, i));
			}
		}
	}
	
	//Creates a BAL network with layer sizes [in_size, h_size, out_size] 
	public BAL(int in_size, int h_size, int out_size) {
		log.println("Creating BAL of size ["+in_size + ","+h_size + ","+out_size + "] RunId=" + NETWORK_RUN_ID);
		//+1 stands for biases 
		this.IH = createInitMatrix(in_size+(isBias(MATRIX_IH)?1:0), h_size);
		this.HO = createInitMatrix(h_size+(isBias(MATRIX_HO)?1:0), out_size);
		this.OH = createInitMatrix(out_size+(isBias(MATRIX_OH)?1:0), h_size);
		this.HI = createInitMatrix(h_size+(isBias(MATRIX_HI)?1:0), in_size);

		// TODO how to cope with biases? 
		if(INIT_SYMMETRIC_IS){
			makeSymmetric(this.OH, this.HO, out_size, h_size);
			makeSymmetric(this.HI, this.IH, h_size, in_size);
		}
		
		this.BAL_construct_other(); 
		
		//System.out.println(this.printNetwork()); 
	}

	@SuppressWarnings("unchecked")
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

		// DROPOUT: mask which hidden layer neurons should be used 
		this.active_hidden = new boolean[h_size + 1]; // +1 for the bias 
		for(int i=0; i<this.active_hidden.length ; i++){
			this.active_hidden[i] = true; 
		}

		all_true_active_hidden = new boolean[Math.max(in_size, Math.max(h_size, out_size)) + 1];
		for(int i=0; i<all_true_active_hidden.length ; i++){
			all_true_active_hidden[i] = true; 
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
		log.println("Creating BAL from file '" + filename + "' RunId=" + NETWORK_RUN_ID);
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

	private boolean isBias(int which_matrix){
		if(WEIGHT_UPDATE_TYPE == GENEREC_WEIGHT_UPDATE){
			if(which_matrix == MATRIX_IH || which_matrix == MATRIX_HO){ //|| which_matrix == MATRIX_OH){
				return true;
			}
			else{
				return false;   
			}
		}
		else{
			return true; 
		}
	}
	
	private RealVector addBias(RealVector in, int which_matrix){
		return isBias(which_matrix) ? in.append(1.0) : in.copy(); 
	}

	private void applyDropoutInPass(RealVector hidden){
		if(BAL.DROPOUT_IS){
			for(int i=0; i<this.active_hidden.length ; i++){
				if(!this.active_hidden[i]){
					hidden.setEntry(i, 0.0);  
				}
			}
		}
	}

	// TODO merge forwardPass with backwardPass
	//forward activations 
	private RealVector[] forwardPass(RealVector in){
		if(INIT_RECIRCULATION_IS){
			return forwardPassWithRecirculation(in); 
		}
		
		RealVector[] forward = new RealVector[3]; 
		forward[0] = addBias(in, MATRIX_IH);

		forward[1] = this.IH.preMultiply(forward[0]);
		applyNonlinearity(forward[1]);
		applyDropoutInPass(forward[1]);
		forward[1] = addBias(forward[1], MATRIX_HO);

		forward[2] = this.HO.preMultiply(forward[1]);
		applyNonlinearity(forward[2]);

		return forward; 
	}

	public static RealVector getAverage(RealVector rv1, RealVector rv2){
		return rv1.add(rv2).mapDivide(2.0);
	}
	
	// TODO merge forwardPassWithRecirculation with backwardPassWithRecirculation
	public static double max_fluctuation = 0.0; 
	public static Set<String> max_fluctuation_run_ids = new HashSet<String>(); 
	private RealVector[] forwardPassWithRecirculation(RealVector in){
		RealVector[] forward = new RealVector[3]; 
		forward[0] = addBias(in, MATRIX_IH);
		forward[2] = new ArrayRealVector(this.HO.getColumnDimension(), 0.0);
		
		RealVector hidden_net_from_input = this.IH.preMultiply(forward[0]);
		RealVector last_hid_activation = forward[1]; 
		RealVector last_out_activation = forward[2]; 
		
		ArrayList<RealVector> h = new ArrayList<RealVector>();

		//TODO when reached max, take average of last two
		int cc = 0; 
		double max_change = 0.0; 
		for(cc=0; cc < RECIRCULATION_ITERATIONS_MAX ; cc++){
			last_hid_activation = forward[1]; 
			last_out_activation = forward[2];
			
			forward[2] = addBias(forward[2], MATRIX_OH); 
			
			RealVector hidden_net_from_output = this.OH.preMultiply(forward[2]);
			
			forward[1] = hidden_net_from_input.add(hidden_net_from_output);
			//forward[1].mapMultiplyToSelf(0.5); //DEVELOPER HALF
			applyNonlinearity(forward[1]);
			//applyDropoutInPass(forward[1]);
			forward[1] = addBias(forward[1], MATRIX_HO); 
			
			forward[2] = this.HO.preMultiply(forward[1]);
			applyNonlinearity(forward[2]);
			
			h.add(forward[1]);
			h.add(forward[2]); 
			
			// stop when no bigger change 
			max_change = 0.0; 
			for(int j=0; j<last_out_activation.getDimension() ; j++) {
				max_change = Math.max(max_change, Math.abs(last_out_activation.getEntry(j) - forward[2].getEntry(j)));
			}
			
			if(cc > 0 && max_change <= RECIRCULATION_EPSILON) {
				break; 
			}
		}
		
		//It's relevant only to monitor activation changes at end of iteration 
		max_fluctuation = Math.max(max_fluctuation, max_change);
		//recirc_iter_counts.add(cc); 
		/*
		System.out.println("max fluctuation: " + max_fluctuation);
		System.out.println("  max: " + max);
		System.out.print("  " + printVector(last_out_activation));
		System.out.print("  " + printVector(forward[2]));
		*/ 

		if(IS_PRINT || max_fluctuation > 0.05 && !max_fluctuation_run_ids.contains(NETWORK_RUN_ID)){
			max_fluctuation_run_ids.add(NETWORK_RUN_ID);
			
			System.out.print("forwardPassWithRecirculation : " + printVector(in));
			System.out.println("  RUN_ID: " + NETWORK_RUN_ID);
			System.out.println("  Epoch:  " + NETWORK_EPOCH);
			System.out.println("  Max fluctuation: " + max_fluctuation);
			System.out.println("  Iteration count: " + cc);
			System.out.println("  Recirc epsilon : " + RECIRCULATION_EPSILON);
			for(int i=2*RECIRCULATION_ITERATIONS_MAX-10; i<h.size() ; i += 2){System.out.print("  Hidden activations: " + printVector(h.get(i)));}
			for(int i=2*RECIRCULATION_ITERATIONS_MAX-10+1; i<h.size() ; i += 2){System.out.print("  Output activations: " + printVector(h.get(i)));}
			System.out.println("Network: " + this.printNetwork());
			System.out.println();
			
		}

		if(cc == RECIRCULATION_ITERATIONS_MAX && RECIRCULATION_USE_AVERAGE_WHEN_OSCILATING){
			forward[1] = getAverage(last_hid_activation, forward[1]); 
			forward[2] = getAverage(last_out_activation, forward[2]); 
		}
		
		return forward; 
	}

	//backward activations 
	private RealVector[] backwardPass(RealVector out){
		if(INIT_RECIRCULATION_IS){
			return backwardPassWithRecirculation(out); 
		}
		
		RealVector[] backward = new RealVector[3]; 
		backward[2] = addBias(out, MATRIX_OH); 

		backward[1] = this.OH.preMultiply(backward[2]);
		applyNonlinearity(backward[1]);
		//applyDropoutInPass(backward[1]);
		backward[1] = addBias(backward[1], MATRIX_HI); 

		backward[0] = this.HI.preMultiply(backward[1]);
		applyNonlinearity(backward[0]);

		return backward; 
	}
	
	//
	private RealVector[] backwardPassWithRecirculation(RealVector out){
		RealVector[] backward = new RealVector[3]; 
		backward[2] = addBias(out, MATRIX_OH);
		backward[0] = new ArrayRealVector(this.HI.getColumnDimension(), 0.0);
		
		RealVector hidden_net_from_output = this.OH.preMultiply(backward[2]);
		RealVector last_in_activation = backward[0]; 
		RealVector last_hid_activation = backward[1]; 
		
		ArrayList<RealVector> h = new ArrayList<RealVector>(); 

		//TODO when reached max, take average of last two 
		int cc = 0; 
		double max_change = 0.0; 
		for(cc=0; cc < RECIRCULATION_ITERATIONS_MAX ; cc++){
			last_in_activation = backward[0];
			last_hid_activation = backward[1];
			
			backward[0] = addBias(backward[0], MATRIX_IH); 
			RealVector hidden_net_from_input = this.IH.preMultiply(backward[0]);
			
			backward[1] = hidden_net_from_input.add(hidden_net_from_output);
			//backward[1].mapMultiplyToSelf(0.5); //DEVELOPER HALF
			applyNonlinearity(backward[1]);
			//applyDropoutInPass(backward[1]);
			backward[1] = addBias(backward[1], MATRIX_HI); 
			
			backward[0] = this.HI.preMultiply(backward[1]);
			applyNonlinearity(backward[0]);
			
			h.add(backward[1]);
			h.add(backward[0]); 

			// stop when no bigger change 
			max_change = 0.0; 
			for(int j=0; j<last_in_activation.getDimension() ; j++) {
			  max_change = Math.max(max_change, Math.abs(last_in_activation.getEntry(j) - backward[0].getEntry(j)));
			}
			if(cc > 0 && max_change <= RECIRCULATION_EPSILON) {
				break; 
			}
		}

		// it's relevant only to monitor activation divergence 
		max_fluctuation = Math.max(max_fluctuation, max_change);
		//recirc_iter_counts.add(cc); 

		if(IS_PRINT || max_fluctuation > 0.05 && !max_fluctuation_run_ids.contains(NETWORK_RUN_ID)){
			max_fluctuation_run_ids.add(NETWORK_RUN_ID);
			
			System.out.print("backwardPassWithRecirculation : " + printVector(out));
			System.out.println("  RUN_ID: " + NETWORK_RUN_ID);
			System.out.println("  Epoch:  " + NETWORK_EPOCH);
			System.out.println("  Max fluctuation: " + max_fluctuation);
			System.out.println("  Iteration count: " + cc);
			System.out.println("  Recirc epsilon : " + RECIRCULATION_EPSILON);
			for(int i=2*RECIRCULATION_ITERATIONS_MAX-10; i<h.size() ; i += 2){System.out.print("  Hidden activations: " + printVector(h.get(i)));}
			for(int i=2*RECIRCULATION_ITERATIONS_MAX-10+1; i<h.size() ; i += 2){System.out.print("  Input activations: " + printVector(h.get(i)));}
			System.out.println("Network: " + this.printNetwork());
			System.out.println();
			
			max_fluctuation = 0.0; 
		}

		if(cc == RECIRCULATION_ITERATIONS_MAX && RECIRCULATION_USE_AVERAGE_WHEN_OSCILATING){
			backward[1] = getAverage(last_hid_activation, backward[1]); 
			backward[0] = getAverage(last_in_activation, backward[0]); 
		}
		
		return backward; 
	}
	
	private RealVector bothwardPass(RealVector in, RealVector out){
		RealVector hidden_net_from_input = this.IH.preMultiply(addBias(in, MATRIX_IH));
		RealVector hidden_net_from_output = this.OH.preMultiply(addBias(out, MATRIX_OH));
		//backward[1].mapMultiplyToSelf(0.5); //DEVELOPER HALF
		RealVector result = hidden_net_from_input.add(hidden_net_from_output);
		applyNonlinearity(result);
		return result; 
	}

	//learns on a weight matrix, other parameters are activations on needed layers 
	//\delta w_{pq}^F = \lambda a_p^{F}(a_q^{B} - a_q^{F})
	//\delta w_ij = lamda * a_pre * (a_post_other - a_post_self)  
	private static void subLearn(RealMatrix w, RealVector a_pre, RealVector a_post_other, RealVector a_post_self, double lambda, double[][] mom, double[][] batch, boolean[] is_train_pre, boolean[] is_train_post){
		for(int i = 0 ; i < w.getRowDimension() ; i++){
			if(!is_train_pre[i]) continue; //dropout 

			for(int j = 0 ; j < w.getColumnDimension() ; j++){
				if(!is_train_post[j]) continue; //dropout

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
			}
		}
	}

	//TODO: dropout 
	//TODO: batch 
	//learns on a weight matrix, other parameters are activations on needed layers 
	//\delta w_{pq}^F = \lambda a_p^{F}(a_q^{B} - a_q^{F})
	//\delta w_ij = lamda * a_pre * (a_post_other - a_post_self)  
	private static void subCHLLearn(RealMatrix w, RealVector a_plus_i, RealVector a_plus_j, RealVector a_minus_j, RealVector a_minus_i, double lambda){
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
			}
		}
	}

	private static void resetTwoDimArray(double[][] arr){
		for(int i=0; i<arr.length ; i++){
			for(int j=0; j<arr[i].length ; j++){
				arr[i][j] = 0.0; 
			}
		}
	}

	private static void updateTwoDimArray(RealMatrix m, double[][] arr){
		for(int i=0; i<arr.length ; i++){
			for(int j=0; j<arr[i].length ; j++){
				m.setEntry(i, j, m.getEntry(i, j) + arr[i][j]); 
			}
		}
	}

	//learn on one input-output mapping
	public void learn(RealVector in, RealVector target, double lambda){
		if(DROPOUT_IS){
			for(int i=0; i<this.active_hidden.length ; i++){
				this.active_hidden[i] = BAL.random.nextBoolean(); 
			}
		}
		boolean[] d_all = BAL.all_true_active_hidden;
		boolean[] d_hidden = this.active_hidden; 

		//learn
		if(WEIGHT_UPDATE_TYPE == BAL_WEIGHT_UPDATE){
			//IS_PRINT = true; 
			RealVector[] forward = this.forwardPass(in);
			RealVector[] backward = this.backwardPass(target);
			//IS_PRINT = false; 
			
			subLearn(this.IH, forward[0], backward[1], forward[1], lambda, this.MOM_IH, this.BATCH_IH, d_all, d_hidden); 
			subLearn(this.HO, forward[1], backward[2], forward[2], lambda, this.MOM_HO, this.BATCH_HO, d_hidden, d_all); 
			subLearn(this.OH, backward[2], forward[1], backward[1], lambda, this.MOM_OH, this.BATCH_OH, d_all, d_hidden); 
			subLearn(this.HI, backward[1], forward[0], backward[0], lambda, this.MOM_HI, this.BATCH_HI, d_hidden, d_all);
		}
		if(WEIGHT_UPDATE_TYPE == BAL_RECIRC_WEIGHT_UPDATE){
			/**/ 
			//IS_PRINT = true; 
			RealVector[] forward = this.forwardPassWithRecirculation(in);
			RealVector[] backward = this.backwardPassWithRecirculation(target);
			//IS_PRINT = false; 
			
			subLearn(this.IH, forward[0], backward[1], forward[1], lambda, this.MOM_IH, this.BATCH_IH, d_all, d_hidden); 
			subLearn(this.HO, forward[1], backward[2], forward[2], lambda, this.MOM_HO, this.BATCH_HO, d_hidden, d_all); 
			
			if(INIT_SYMMETRIC_IS){
				makeSymmetric(this.HI, this.IH, this.IH.getColumnDimension(), this.IH.getRowDimension() - (isBias(MATRIX_IH)?1:0));
				makeSymmetric(this.OH, this.HO, this.HO.getColumnDimension(), this.HO.getRowDimension() - (isBias(MATRIX_HO)?1:0));
			}
			
			subLearn(this.OH, backward[2], forward[1], backward[1], lambda, this.MOM_OH, this.BATCH_OH, d_all, d_hidden); 
			subLearn(this.HI, backward[1], forward[0], backward[0], lambda, this.MOM_HI, this.BATCH_HI, d_hidden, d_all);
			
			if(INIT_SYMMETRIC_IS){
				makeSymmetric(this.IH, this.HI, this.HI.getColumnDimension(), this.HI.getRowDimension() - (isBias(MATRIX_HI)?1:0));
				makeSymmetric(this.HO, this.OH, this.OH.getColumnDimension(), this.OH.getRowDimension() - (isBias(MATRIX_OH)?1:0));
			}
			/*/
			//TODO not working - zero activations on INPUT / OUTPUT 
			//IS_PRINT = true; 
			RealVector[] forward = this.forwardPassWithRecirculation(in);
			RealVector[] backward = this.backwardPassWithRecirculation(target);
			RealVector bothward = this.bothwardPass(in, target); 
			//IS_PRINT = false; 
			
			subLearn(this.IH, forward[0], bothward, forward[1], lambda, this.MOM_IH, this.BATCH_IH, d_all, d_hidden); 
			subLearn(this.HO, forward[1], backward[2], forward[2], lambda, this.MOM_HO, this.BATCH_HO, d_hidden, d_all); 
			
			if(INIT_SYMMETRIC_IS){
				makeSymmetric(this.HI, this.IH, this.IH.getColumnDimension(), this.IH.getRowDimension() - (isBias(MATRIX_IH)?1:0));
				makeSymmetric(this.OH, this.HO, this.HO.getColumnDimension(), this.HO.getRowDimension() - (isBias(MATRIX_HO)?1:0));
			}
			
			subLearn(this.OH, backward[2], bothward, backward[1], lambda, this.MOM_OH, this.BATCH_OH, d_all, d_hidden); 
			subLearn(this.HI, backward[1], forward[0], backward[0], lambda, this.MOM_HI, this.BATCH_HI, d_hidden, d_all);
			
			if(INIT_SYMMETRIC_IS){
				makeSymmetric(this.IH, this.HI, this.HI.getColumnDimension(), this.HI.getRowDimension() - (isBias(MATRIX_HI)?1:0));
				makeSymmetric(this.HO, this.OH, this.OH.getColumnDimension(), this.OH.getRowDimension() - (isBias(MATRIX_OH)?1:0));
			}
			/**/
		}
		if(WEIGHT_UPDATE_TYPE == GENEREC_WEIGHT_UPDATE) {
			//symmetric version 
			//IS_PRINT = true; 
			RealVector[] forward = this.forwardPassWithRecirculation(in); // TODO HO = OH 
			RealVector bothward = this.bothwardPass(in, target); 
			//RealVector biased_target = addBias(target); 
			//IS_PRINT = false; 
			
			//TODO why biased target? 
			subLearn(this.IH, forward[0], bothward, forward[1], lambda, this.MOM_IH, this.BATCH_IH, d_all, d_hidden); 
			subLearn(this.HO, forward[1], target, forward[2], lambda, this.MOM_HO, this.BATCH_HO, d_hidden, d_all); 
			//subLearn(this.OH, biased_target, forward[1], bothward, lambda, this.MOM_OH, this.BATCH_OH, d_all, d_hidden); 
			
			makeSymmetric(this.OH, this.HO, this.HO.getColumnDimension(), this.HO.getRowDimension() - (isBias(MATRIX_HO)?1:0));
			//makeSymmetric(this.HI, this.IH, this.HI.getRowDimension(), this.HI.getColumnDimension());
			
			/*
			System.out.print("Input: " + printVector(in));
			System.out.println("Forward pass:");
			for(int i=0; i<3; i++){
				System.out.print("  " + printVector(forward[i]));
			}
			System.out.println("Bothward: " + printVector(bothward));
			System.out.println("Network:\n" + printNetwork()); */ 
		}
		if(WEIGHT_UPDATE_TYPE == CHL_WEIGHT_UPDATE){
			RealVector[] forward = this.forwardPass(in);
			RealVector[] backward = this.backwardPass(target);
			subCHLLearn(this.IH, forward[0], forward[1], backward[1], backward[0], lambda);
			subCHLLearn(this.HO, forward[1], forward[2], backward[2], backward[1], lambda);
			subCHLLearn(this.OH, backward[2], backward[1], forward[1], forward[2], lambda);
			subCHLLearn(this.HI, backward[1], backward[0], forward[0], forward[1], lambda);
		}

		//log.print(BAL.printVector(forward[1]));
		//log.println(BAL.printVector(backward[1]));
	}

	public double error(RealVector given_activation, RealVector target){
		BAL.postprocessOutput(given_activation);

		double error = 0.0; 
		for(int i=0; i<target.getDimension() ; i++){
			error += Math.pow(given_activation.getEntry(i) - target.getEntry(i), 2); 
		}

		return error;  
	}
	
	//evaluates performance on one input-output mapping 
	//returns error 
	public double evaluate(RealVector in, RealVector target){
		RealVector[] forward = forwardPass(in);
		return this.error(forward[forward.length - 1], target); 
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

	//collect monitoring=measure data, epoch is used as identifier
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

				if(forward[1].getDimension() == backward[1].getDimension()){
					hidden_for_back_dist += forward[1].getDistance(backward[1]) / n;
				}
				if(forward[2].getDimension() == backward[0].getDimension()){
					output_for_back_dist += forward[2].getDistance(backward[0]) / n;
				}

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

				/*//DEVELOPER DEBUG 
				log.println("Hidden points");
				log.println(hidden_points);
				log.println("ConvexHull points");
				log.println(convex_hull);
				log.println("End"); */
			}
			
		}

		if(MEASURE_FLUCTUATION < MEASURE_COUNT){
			max_fluctuation = 0.0; 
			for(int i=0; i<in.getRowDimension(); i++){
				this.forwardPassWithRecirculation(in.getRowVector(i));
				this.backwardPassWithRecirculation(target.getRowVector(i));
			}
			this.measures[MEASURE_FLUCTUATION].add(max_fluctuation); 
			max_fluctuation = 0.0; 
		}
		
		if(MEASURE_MATRIX_AVG_W < MEASURE_COUNT){
			double matrix_avg_w = 0.0;
			matrix_avg_w = (sumAbsoluteValuesOfMatrixEntries(this.IH) + sumAbsoluteValuesOfMatrixEntries(this.HO) + sumAbsoluteValuesOfMatrixEntries(this.OH) + sumAbsoluteValuesOfMatrixEntries(this.IH)) / (this.IH.getColumnDimension()*this.IH.getRowDimension() + this.HO.getColumnDimension()*this.HO.getRowDimension()+ this.OH.getColumnDimension()*this.OH.getRowDimension()+ this.HI.getColumnDimension()*this.HI.getRowDimension()); 
			this.measures[MEASURE_MATRIX_AVG_W].add(matrix_avg_w);
		}

		if(MEASURE_MATRIX_SIMILARITY < MEASURE_COUNT){
			double matrix_similarity = 0.0;
			if(MEASURE_MATRIX_SIMILARITY >= 0 && this.HO.getColumnDimension() == this.HI.getColumnDimension() && this.HO.getRowDimension() == this.HI.getRowDimension()
					&& this.OH.getColumnDimension() == this.IH.getColumnDimension() && this.OH.getRowDimension() == this.IH.getRowDimension()){
				RealMatrix diff_HO_HI = this.HO.subtract(this.HI); 
				RealMatrix diff_OH_IH = this.OH.subtract(this.IH);
				matrix_similarity = (sumAbsoluteValuesOfMatrixEntries(diff_HO_HI) + sumAbsoluteValuesOfMatrixEntries(diff_OH_IH)) / (this.IH.getColumnDimension()*this.IH.getRowDimension() + this.HI.getColumnDimension()*this.HI.getRowDimension());
			}   
			this.measures[MEASURE_MATRIX_SIMILARITY].add(matrix_similarity);
		}

		double[] result = new double[MEASURE_COUNT]; 
		for(int i=0; i<MEASURE_COUNT; i++){
			result[i] = this.measures[i].get(this.measures[i].size()-1); 
		}
		return result; 
	}

	// TODO comment it 
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

	// TODO java.String.join ? 
	public static String printVector(RealVector v){
		StringBuilder sb = new StringBuilder();

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
	public boolean saveMeasures(String run_id, PrintWriter writer){
		if(!MEASURE_IS) return true; 

		double[] m = new double[MEASURE_COUNT];
		for(int j=0; j<MEASURE_COUNT; j++){
			m[j] = 1;
		}

		//writer.println(this.measures[0].size() + " " + this.measures.length);
		writer.write("RUN_ID");
		for(int i=0; i<MEASURE_HEADINGS.length ; i++){
			writer.write("\t");
			writer.write(MEASURE_HEADINGS[i]); 
		}
		writer.println(); 

		for(int i=0; i<this.measures[0].size(); i++){
			writer.write(run_id);
			for(int j=0; j<this.measures.length; j++){
				writer.write("\t"); 
				writer.print(this.measures[j].get(i) / m[j]); 
			}
			writer.println(); 
		}

		return true; 
	}	

	//TODO Refactor with saveMeasures()
	public static void printPreAndPostMeasures(String global_run_id){
		if(!MEASURE_IS) return; 

		PrintWriter pre_writer;
		PrintWriter post_writer;
		try {
			pre_writer = new PrintWriter("data/" + global_run_id + "_pre.csv", "UTF-8");
			post_writer = new PrintWriter("data/" + global_run_id + "_post.csv", "UTF-8");
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

		log.println("PreMeasure : GroupBy");
		System.out.println("PreMeasure : GroupBy");
		BAL.measureGroupBY(pre_measure); 
		log.println("PostMeasure : GroupBy");
		System.out.println("PostMeasure : GroupBy");
		BAL.measureGroupBY(post_measure);
		
		log.println("PreMeasure : Averages");
		BAL.measureAverages(pre_measure); 
		log.println("PostMeasure : Averages");
		BAL.measureAverages(post_measure); 
	}

	// based on results it saves the network to "good" / "bad" folders 
	public static void printHiddenRepresentations() throws FileNotFoundException, UnsupportedEncodingException{
		if(!HIDDEN_REPRESENTATION_IS){
			return;
		}

		for(int i=0; i<hidden_repre_all.size() ; i++){
			ArrayList<RealVector[]> priebeh = hidden_repre_all.get(i);

			for(int k=0; k < priebeh.get(0).length ; k++){
				String filename = HIDDEN_REPRESENTATION_DIRECTORY + ((post_measure.get(i)[MEASURE_ERROR] == 0.0) ? "good" : "bad") + "/" + MEASURE_RUN_ID.get(i) + "_" + k + ".csv";
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

	public static void generateNetworkRunId(){
		NETWORK_RUN_ID = INPUT_FILEPATH.substring(0, INPUT_FILEPATH.indexOf('.')) + "_" + (System.currentTimeMillis()) + "_" + INIT_HIDDEN_LAYER_SIZE;
	}

	public static String experimentInit() throws IOException{
		generateNetworkRunId(); 
		String global_run_id = NETWORK_RUN_ID; // it changes after each new network generation 

		log = new PrintWriter("data/" + global_run_id + ".log");

		measure_writer = new PrintWriter("data/" + global_run_id + "_measure.csv");
		pre_measure = new ArrayList<double[]>();
		post_measure = new ArrayList<double[]>();

		if(HIDDEN_REPRESENTATION_IS){
			hidden_repre_all = new ArrayList<ArrayList<RealVector[]>>(); 
		}

		return global_run_id; 
	}

	public static void experimentRun(BAL network) throws FileNotFoundException {
		for(int i=0 ; i<BAL.INIT_RUNS ; i++){
			long start_time = System.currentTimeMillis(); 

			log.println("  ======== " + i + "/" + BAL.INIT_RUNS + " ==============");
			System.out.println("  ======== " + i + "/" + BAL.INIT_RUNS + " ==============");

			@SuppressWarnings("unused")
			Double error = BAL.run(network); 

			long run_time = (System.currentTimeMillis() - start_time); 
			log.println("RunTime=" + run_time);
			System.out.println("RunTime=" + run_time);
		}
	}

	public static void experimentFinalize(String global_run_id) throws FileNotFoundException, UnsupportedEncodingException {
		printPreAndPostMeasures(global_run_id);
		printHiddenRepresentations(); 
		log.close(); 
		measure_writer.close(); 
	}

	public static void experiment_Default() throws IOException{
		String global_run_id = experimentInit();
		experimentRun(null);
		experimentFinalize(global_run_id);
		
		if(!recirc_iter_counts.isEmpty()){
			int sum=0; 
			for(int i=0; i<recirc_iter_counts.size(); i++) {
					System.out.println(recirc_iter_counts.get(i));
					sum += recirc_iter_counts.get(i); 
			}
			System.out.println("Iteration recirc avg=" + (sum / recirc_iter_counts.size()) + " sum=" + sum + " count=" + recirc_iter_counts.size());
		}
	}

	// TODO test 
	public static void experiment_RerunGoodBad() throws IOException{
		String global_run_id = experimentInit(); 

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

			BAL network = new BAL(filepath); 			
			experimentRun(network);
		}

		experimentFinalize(global_run_id);
	}

	// TODO test 
	public static void experiment_DifferentHiddenSizes(String input_prefix) throws IOException{
		MEASURE_IS = true; 
		MEASURE_SAVE_AFTER_EACH_RUN = false; 
		MEASURE_RECORD_EACH = 50;

		INPUT_FILEPATH = input_prefix + ".in"; 
		OUTPUT_FILEPATH = input_prefix + ".out"; 

		CONVERGENCE_NO_CHANGE_FOR = 10; 
		INIT_MAX_EPOCHS = 10000;

		INIT_RUNS = 1; 
		INIT_CANDIDATES_COUNT = 0;

		PRINT_NETWORK_IS = false; 

		TRY_NORMAL_DISTRIBUTION_SIGMA = new double[] {1.0}; 
		TRY_LAMBDA = new double[] {0.2}; 
		TRY_MOMENTUM = new double[] {0.0}; 

		for(int h=5; h<=144; h += h/8 + 1){
			INIT_HIDDEN_LAYER_SIZE = h; 
			experiment_Default();
		} 
	}

	public static void experiment_TestImplementation() throws IOException{
		MEASURE_IS = true; 
		MEASURE_SAVE_AFTER_EACH_RUN = true; 
		MEASURE_RECORD_EACH = 100;

		INPUT_FILEPATH = "auto4.in"; 
		OUTPUT_FILEPATH = "auto4.in"; 
		INIT_HIDDEN_LAYER_SIZE = 2; 

		INIT_NORMAL_DISTRIBUTION_SIGMA = 2.3;  
		INIT_LAMBDA = 0.7; 
		INIT_MAX_EPOCHS = 10000;
		INIT_RUNS = 100; 
		INIT_CANDIDATES_COUNT = 1;
		INIT_SHUFFLE_IS = false;
		INIT_BATCH_IS = false;
		INIT_SYMMETRIC_IS = false; 	
		
		RECIRCULATION_EPSILON = 0.001; //if the max unit activation change is less the RECIRCULATION_EPSILON, it will stop 
		RECIRCULATION_ITERATIONS_MAX = 200; //maximum number of iterations to approximate the underlying dynamic system  
		RECIRCULATION_USE_AVERAGE_WHEN_OSCILATING = true;

		DROPOUT_IS = false; 
		CONVERGENCE_NO_CHANGE_FOR = INIT_MAX_EPOCHS; 

		INIT_MOMENTUM_IS = false; 
		
		INIT_NORMAL_DISTRIBUTION_MU = 0;
		NORMAL_DISTRIBUTION_SPAN = 15; 
 
		HIDDEN_REPRESENTATION_IS = false;
		HIDDEN_REPRESENTATION_DIRECTORY = "data/test/"; 
		HIDDEN_REPRESENTATION_EACH = 1; 
		HIDDEN_REPRESENTATION_AFTER = 200;
		HIDDEN_REPRESENTATION_ONLY_EACH = 200;

		PRINT_NETWORK_IS = false;  
		
		experiment_Default();
	}
}
