// ============== MUST =================
//TODO be sure with postprocesser output! 
//TODO experiment pozostavajuci zo skumanych modelov a ich performance na danych troch datasetoch
//TODO speatna reprezentacia hand-written 
//TODO original BAL - epochs don't correspond to the Farkas, Rebrova article

// ============== NICE TO HAVE =========
//TODO pridat deliace ciary do hidden activation (HI)
//TODO rekonstrukcia (zmena zopar bitov, ci tam-speat da orig)
//TODO noise
//TODO speed up by only calculating what is necessary
//TODO have a C++ implementation for fastest possible network
//  TODO comment the final (i.e. best) solution and make it reusable (could be after deadline, helps on presentation)   

// ============== OPTIONAL =============
//TODO dynamicka rychlost ucenia 
//TODO MEASURE activation change   
//TODO dropout 
//TODO kvazi momentum -> odtlacanie hidden reprezentacii, -\delta w(t-1)
//TODO spektralna analyza

/**
 * Author: Peter Csiba
 * Email: petherz@gmail.com 
 * Github: https://github.com/Petrzlen/master_thesis
 * License: Share-alike with author name, email, github
 */

import java.awt.Point;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class BAL {
	//manage IO and run BAL 
	public static void main(String[] args) throws IOException {
		DECIMAL_FORMAT.setMaximumFractionDigits(9);

		//experiment_Default();
		//experiment_DifferentHiddenSizes("k3");
		//experiment_RerunGoodBad();
		experiment_TestImplementation();
		//experiment_Digits(); 
	}

	private static DecimalFormat DECIMAL_FORMAT = new DecimalFormat("0"); 

	private static boolean IS_PRINT = false; 

	//TODO symetricka verzia so standardnym BALom 
	private static int WU_BAL_ORIG = 1; 
	private static int WU_GENEREC_CHL = 2; // works but slow 
	private static int WU_BAL_RECIRC = 3; 
	private static int WU_GENEREC = 4; // => INIT_SYMMETRIC_IS = true
	private static int WU_GENEREC_SYM = 5; // => INIT_SYMMETRIC_IS = true
	private static int WU_GENEREC_MID = 6; // => INIT_SYMMETRIC_IS = true 
	private static int WU_BAL_SYM = 7; // non of BAL other learning rule works 
	private static int WU_BAL_MID = 8; 
	private static int WU_BAL_CHL = 9; 
	private static final int WU_TYPE = WU_BAL_RECIRC;

	public static final boolean INIT_SYMMETRIC_IS = (WU_TYPE == WU_GENEREC || WU_TYPE == WU_GENEREC_CHL || 
			WU_TYPE == WU_GENEREC_MID || WU_TYPE == WU_GENEREC_SYM);
	
	// ========= RECIRCULATION -- iterative activation ==============
	private static boolean INIT_RECIRCULATION_IS = (WU_TYPE == WU_BAL_RECIRC || WU_TYPE == WU_GENEREC || WU_TYPE == WU_GENEREC_SYM || WU_TYPE == WU_GENEREC_MID);
	//if the max unit activation change is less the RECIRCULATION_EPSILON, it will stop
	private static double RECIRCULATION_EPSILON = 0.01; 
	//maximum number of iterations to approximate the underlying dynamic system
	private static int RECIRCULATION_ITERATIONS_MAX = 200; 
	 // average of last two activations will be used instead of the last one (intuition: more stable)
	private static boolean RECIRCULATION_USE_AVERAGE_WHEN_OSCILATING = true; 

	private static PrintWriter log = null; 

	public static double DOUBLE_EPSILON = 0.001;

	public static boolean MEASURE_IS = true; 
	public static boolean MEASURE_SAVE_AFTER_EACH_RUN = true; 
	public static int MEASURE_RECORD_EACH = 100;
	public static boolean POSTPROCESS_INPUT = false; // i.e. if treshold should be applied on the input layer (-> 0.0 or 1.0 result)  
	public static boolean POSTPROCESS_OUTPUT = true; // i.e. if treshold should be applied on the output layer (-> 0.0 or 1.0 result)
	public static final int POSTPROCESS_SIMPLE = 0; 
	public static final int POSTPROCESS_MAXIMUM = 1;
	public static int POSTPROCESS_TYPE = POSTPROCESS_SIMPLE;  
	
	public static String INPUT_FILEPATH = "auto4.in"; 
	public static String OUTPUT_FILEPATH = "auto4.in"; 
	public static String TEST_INPUT_FILEPATH = null; 
	public static String TEST_OUTPUT_FILEPATH = null; 
	public static int INIT_HIDDEN_LAYER_SIZE = 2 ; 

	public static int INIT_MAX_EPOCHS = 500000;
	public static int INIT_RUNS = 1000; 
	public static int INIT_CANDIDATES_COUNT = 1000;
	public static boolean INIT_SHUFFLE_IS = true;
	public static boolean INIT_BATCH_IS = false;
	public static boolean INIT_TRAIN_ONLY_ON_ERROR = false; // network is trained only on samples which give error 

	//which hidden layer neurons are active (bias is not counted), used for dropout
	public static boolean DROPOUT_IS = false; //TODO check some runs, it gives error 
	private boolean[] active_hidden; 
	private static boolean[] all_true_active_hidden; // a mock which says "all hidden units are active" 

	//the network will stop training if no change in result occurred in last CONVERGENCE_NO_CHANGE_FOR
	//  -1 means not applicable 
	public static int CONVERGENCE_NO_CHANGE_FOR = -1; 
	//if true network will stop training if bitSucc^F = 1.0 
	public static boolean STOP_IF_NO_ERROR = true; 
	//(on each measure) remembers best epoch in terms of error and compares to current, if not better then stops  
	//  -1 means not applicable 
	public static int STOP_IF_NO_IMPROVE_FOR = -1;
	public static double STOP_IF_NO_IMPROVE_BEST_ERR = 1987654321;
	public static double STOP_IF_NO_IMPROVE_BEST_EPC = 0; 

	public static double INIT_SIGMA = 2.3; // should be 1 / sqrt{input.size + 1}
	public static double TRY_SIGMA[] = {2.3}; 

	public static double INIT_LAMBDA = 0.7; 
	//public static double TRY_LAMBDA[] = {0.7}; 
	public static double TRY_LAMBDA[] = {0.7}; 

	private static double INIT_LAMBDA_V = 0.0001;
	private static double TRY_LAMBDA_V[] = {0.0001};
	//public static double INIT_LAMBDA_V[] = {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0};
	/*public static double INIT_LAMBDA_V[] = {0.0000001, 0.0000002, 0.0000005, 0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 
											 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 
											 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 
											 100.0}; */
	//public static double INIT_LAMBDA_V[] = {0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005};

	//public static double TRY_NOISE_SPAN[] = {0.0, 0.003, 0.01, 0.03, 0.1, 0.3}; 
	//public static double TRY_MULTIPLY_WEIGHTS[] = {1.0, 1.00001, 1.00003, 1.0001, 1.0003, 1.001}; 

	public static boolean INIT_MOMENTUM_IS = false;  // a performance flag 
	public static double INIT_MOMENTUM = 0.0;  
	public static double TRY_MOMENTUM[] = {0.0};
	//public static double TRY_MOMENTUM[] = {-1, -0.3, -0.1, -0.03, -0.01, -0.003, -0.001, 0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}; 

	// for purposes of dynamic lambda 
	private static boolean LAMBDA_ERROR_MOMENTUM_IS = false; 
	
	public static double INIT_NORMAL_DISTRIBUTION_MU = 0;
	public static double NORMAL_DISTRIBUTION_SPAN = 15; 

	// save the hidden representations 
	public static boolean HIDDEN_REPRESENTATION_IS = false;
	public static String HIDDEN_REPRESENTATION_DIRECTORY = "data/hr/"; 
	public static int HIDDEN_REPRESENTATION_EACH = 1; 
	public static int HIDDEN_REPRESENTATION_AFTER = 200;
	public static int HIDDEN_REPRESENTATION_ONLY_EACH = 50;

	public static boolean PRINT_NETWORK_IS = false;
	//Dump each network which was trained 
	//NOTE: significantly decreases performance
	public static boolean PRINT_NETWORK_TO_FILE_IS = false;  
	
	public static boolean PRINT_EPOCH_SUMMARY = false; 

	//=======================  "TELEMETRY" of the network in time =========================== 
	public static Map<Integer, String> MEASURE_RUN_ID = new HashMap<Integer, String>(); 
	public static long MEASURE_EXECUTION_TIME = 0; 
	public static String[] MEASURE_HEADINGS = {
		"epoch", "err", "lam", "lam_v", "mom", "sigma",  
		"bs_f", "bs_b", "ps_f", "ps_b", 
		"m_wei", "m_sim", "h_fb_d", "o_fb_d", 
		"h_dist", "in_tri", "fluct"
	};
	public static int MEASURE_EPOCH = 0;
	public static int MEASURE_ERROR = 1; //error function (RMSE), bitSucc forward 
	public static int MEASURE_LAMBDA = 2; 
	public static int MEASURE_LAMBDA_V = 3; 
	public static int MEASURE_MOMENTUM = 4;
	public static int MEASURE_SIGMA = 5; 

	public static int MEASURE_BITSUCC_FORWARD = 6;  

	public static int MEASURE_BITSUCC_BACKWARD = 7;  

	public static int MEASURE_PATSUCC_FORWARD = 8;  

	public static int MEASURE_PATSUCC_BACKWARD = 9; 

	//avg weight of matrixes 
	public static int MEASURE_MATRIX_AVG_W = 10;

	//sum of |a_{ij} - b_{ij}| per pairs (HO, HI) and (OH, IH) 
	public static int MEASURE_MATRIX_SIMILARITY = 11;


	//avg distance between forward and backward activations on hidden layer
	public static int MEASURE_HIDDEN_FOR_BACK_DIST = 12;
	
	//avg distance between forward and backward activations on their output layers (forward layer 2, backward layer 0) 
	//  NOTE: could be irrelevant: has meaning only for auto associative tasks 
	public static int MEASURE_OUTPUT_FOR_BACK_DIST = 13;
	
	//avg of dist(h_i - h_j) i \neq j where h_i is a hidden activation for input i
	//intuitively: internal representation difference 
	//  NOTE: could be expensive: O(epoch + inputs.size ^ 2 * hidden.size) 
	public static int MEASURE_HIDDEN_DIST = 14;
	
	//check if some point is inside a polygon from others 
	//  NOTE: could be irrelevant: has meaning only for hidden size = 2
	public static int MEASURE_IN_TRIANGLE = 15;
	
	//how much differ activations when iterative method is used
	//  NOTE: could be expensive: O(epoch * RECIRCULATION_ITERATIONS_MAX)  
	public static int MEASURE_FLUCTUATION = 16; 

	public static int MEASURE_COUNT = 17;  

	//public static int[] MEASURE_GROUP_BY_COLS = {MEASURE_ERROR, MEASURE_SIGMA, MEASURE_LAMBDA, MEASURE_IN_TRIANGLE};
	public static int[] MEASURE_GROUP_BY_COLS = {MEASURE_ERROR, MEASURE_SIGMA, MEASURE_LAMBDA, MEASURE_LAMBDA_V, MEASURE_MOMENTUM};

	public static int MEASURE_GROUP_BY = MEASURE_ERROR;  

	// ================= DATA COLLECTORS ===================
	public static Random random = new Random(); 
	//public static double INIT_NOISE_SPAN = 0.00; 
	//public static double INIT_MULTIPLY_WEIGHTS = 1.001; 

	private ArrayList<Double>[] measures = null; 

	public static ArrayList<double[]> pre_measure = null;
	public static ArrayList<double[]> post_measure = null; 
	public static ArrayList<double[]> test_measure = null; 
	public static PrintWriter measure_writer; 

	// TIMELINE of hidden representations 
	public static ArrayList<ArrayList<RealVector[]>> hidden_repre_all = null;
	public static ArrayList<RealVector[]> hidden_repre_cur = null; 

	//public static ArrayList<Integer> recirc_iter_counts = new ArrayList<Integer>(); 
	//public static ArrayList<Integer> epochs_needed_to_no_error = new ArrayList<Integer>(); 

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

	//Last error matrices 
	private double[][] ERR_IH; 
	private double[][] ERR_HO;  
	private double[][] ERR_OH; 
	private double[][] ERR_HI;

	//Batch matrices (sum all weight changes and update after the current epoch ended) 
	private double[][] BATCH_IH; 
	private double[][] BATCH_HO;  
	private double[][] BATCH_OH; 
	private double[][] BATCH_HI; 

	private static String NETWORK_RUN_ID = null;
	private static int NETWORK_EPOCH = 0; 

	private static boolean isMeasureAtEpoch(int epochs){
		return MEASURE_IS && (MEASURE_SAVE_AFTER_EACH_RUN && epochs % BAL.MEASURE_RECORD_EACH == 0); 
	}
	
	public static BAL getCandidateNetwork(RealMatrix inputs, RealMatrix outputs){
		//select the "best" candidate network
		double mav=0.0; 
		double in_points_best = 1000.0;

		BAL network = new BAL(inputs.getColumnDimension(), BAL.INIT_HIDDEN_LAYER_SIZE, outputs.getColumnDimension()); 
		for(int i=0; i<BAL.INIT_CANDIDATES_COUNT; i++){
			BAL N = new BAL(inputs.getColumnDimension(), BAL.INIT_HIDDEN_LAYER_SIZE, outputs.getColumnDimension()); 
			double[] measure = N.measure(0, inputs, outputs, false); 
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
		
		return network;
	}
	
	public static void printBoth(String s){
		System.out.print(s);
		log.print(s);
	}
	
	// if override_network != null then the provided netwoek will be used (usually loaded from file) 
	public static BAL run(BAL override_network, RealMatrix inputs, RealMatrix outputs) throws IOException{
		NETWORK_EPOCH = 0; 
		max_fluctuation = 0.0; 

		generateNetworkRunId(); 

		if(HIDDEN_REPRESENTATION_IS){
			hidden_repre_cur = new ArrayList<RealVector[]>();
		}

		BAL network = (override_network != null) ? override_network : getCandidateNetwork(inputs, outputs);

		if(PRINT_NETWORK_IS){
			printBoth("----------Network before run: --------------\n");
			printBoth(network.printNetwork());
			printBoth("\n");
		}

		if(PRINT_NETWORK_TO_FILE_IS){
			PrintWriter pw = new PrintWriter("data/networks/" + NETWORK_RUN_ID + "_pre.bal"); 
			pw.write(network.printNetwork());
			pw.close(); 
		}

		if(MEASURE_IS) { 
			MEASURE_RUN_ID.put(pre_measure.size(), NETWORK_RUN_ID);
			pre_measure.add(network.measure(0, inputs, outputs, true));
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
		int current_epoch=1;
		boolean isStop = false; 
		for(current_epoch=1; current_epoch<=BAL.INIT_MAX_EPOCHS ; current_epoch++){
			long st_epoch = System.currentTimeMillis(); 
			if(PRINT_EPOCH_SUMMARY) printBoth("==Running epoch " + current_epoch + "\n"); 
			
			NETWORK_EPOCH = current_epoch; 

			// which hidden representations should be saved 
			RealVector[] hidden_representation = null; 
			boolean is_save_hidden_representation = HIDDEN_REPRESENTATION_IS && ((current_epoch < HIDDEN_REPRESENTATION_AFTER) 
					? current_epoch % HIDDEN_REPRESENTATION_EACH == 0 
					: current_epoch % HIDDEN_REPRESENTATION_ONLY_EACH == 0); 
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

			// check if different outputs given at the last epoch 
			boolean is_different_output = false; 

			// learn on each given / target mapping
			// O(runs * input.size)
			for(int order_cc = 0; order_cc < order.size() ; order_cc++){
				if(PRINT_EPOCH_SUMMARY && (order_cc + 1) % 1000 == 0) printBoth("  input " + (order_cc + 1) + "/" + order.size() + " time=" + (System.currentTimeMillis() - st_epoch) + "\n");
					
				int order_i = order.get(order_cc);
				RealVector in = inputs.getRowVector(order_i);
				RealVector out = outputs.getRowVector(order_i);

				//boolean print_state_is = epochs > INIT_MAX_EPOCHS - 6 && network.evaluate(in, out) > 0.0;
				boolean print_state_is = false;  
				if(print_state_is){
					printBoth("==== error " + order_i + " " + current_epoch + "\n");
					printForwardPass(network.forwardPass(in), out);
					printBackwardPass(network.backwardPass(out));
				}

				network.learn(in, out);

				if(print_state_is){
					log.print("Momentum\n" + network.printMomentum() + "\n");
				}

				if(is_save_hidden_representation){
					RealVector[] forwardPass = network.forwardPass(in); 
					hidden_representation[order_i] = forwardPass[1]; 
				}

				//check if change
				/*
				if(POSTPROCESS_OUTPUT) BAL.postprocessOutput(forwardPass[2]);
				is_diffent_output = is_diffent_output || (last_outputs[order_i].getDistance(forwardPass[2]) > DOUBLE_EPSILON);  
				last_outputs[order_i] = forwardPass[2]; */
			}

			/*
			if(epochs > INIT_MAX_EPOCHS - 20 ){
				System.out.println(epochs + ": " + network.evaluate(inputs, outputs));
			}*/ 

			if(INIT_BATCH_IS){
				updateTwoDimArray(network.IH, network.BATCH_IH);
				updateTwoDimArray(network.HO, network.BATCH_HO);
				updateTwoDimArray(network.OH, network.BATCH_OH);
				updateTwoDimArray(network.HI, network.BATCH_HI);
			}

			if(is_save_hidden_representation){
				hidden_repre_cur.add(hidden_representation); 
			}

			//first measure saved by PRE_MEASURE
			if(isMeasureAtEpoch(current_epoch)){
				//long st_measure = System.currentTimeMillis(); 
				//if(PRINT_EPOCH_SUMMARY) printBoth("  measureStart\n");
				//log.println(network.evaluate(inputs, outputs));
				double[] stats = network.measure(current_epoch, inputs, outputs, true);
				if(PRINT_EPOCH_SUMMARY) {
					//printBoth("  measureEnd time=" + (System.currentTimeMillis() - st_measure) + "\n");
					printBoth("current bitsucc=" + stats[MEASURE_BITSUCC_FORWARD] + "\n");
					printBoth("current patsucc=" + stats[MEASURE_PATSUCC_FORWARD] + "\n");
				}
				
				if(STOP_IF_NO_IMPROVE_FOR >= 0 && STOP_IF_NO_IMPROVE_BEST_ERR > stats[MEASURE_ERROR]){
					STOP_IF_NO_IMPROVE_BEST_ERR = stats[MEASURE_ERROR];
					STOP_IF_NO_IMPROVE_BEST_EPC = current_epoch; 
				}
			}
			
			isStop = false; 
			// no output change for CONVERGENCE_NO_CHANGE_FOR
			no_change_epochs = (is_different_output) ? 0 : no_change_epochs + 1; 
			if(CONVERGENCE_NO_CHANGE_FOR >= 0 && no_change_epochs >= CONVERGENCE_NO_CHANGE_FOR){
				printBoth("Training stopped at epoch=" + current_epoch + " as no output change occured in last " + CONVERGENCE_NO_CHANGE_FOR + "epochs\n");
				isStop = true; 
			}

			// we need to evaluate the performance on each input / output as when non-batch learning the total_error could be changed after weight change 
			if(STOP_IF_NO_ERROR && network.evaluateForward(inputs, outputs) == 0.0){
				//epochs_needed_to_no_error.add(current_epoch); 
				printBoth("Training stopped at epoch=" + current_epoch + " as all outputs given correctly\n");
				isStop = true; 
			}
			
			if(STOP_IF_NO_IMPROVE_FOR >= 0 && (current_epoch - STOP_IF_NO_IMPROVE_BEST_EPC) >= STOP_IF_NO_IMPROVE_FOR){
				printBoth("Training stopped at epoch=" + current_epoch + " as no improvement occured for " + STOP_IF_NO_IMPROVE_FOR + " epochs\n  Best=" + STOP_IF_NO_IMPROVE_BEST_ERR + " in epoch=" + STOP_IF_NO_IMPROVE_BEST_EPC + "\n");
				isStop = true; 
			}

			if(isStop){
				if(MEASURE_IS){
					double[] measure_padding = network.measure(current_epoch, inputs, outputs, false);
					for(int e = current_epoch + 1; e <= BAL.INIT_MAX_EPOCHS ; e++){
						if(isMeasureAtEpoch(e)){ 
							double[] m = measure_padding.clone(); 
							m[MEASURE_EPOCH] = e; 
							network.addMeasure(m);
						}
					}
				}
				break; 
			}

			/*
			if(epochs > INIT_MAX_EPOCHS - 10){
				printNetworkWithPass(network, inputs, outputs, "Network close to end run");
			}*/
			
			if(PRINT_EPOCH_SUMMARY) printBoth("==Epoch End time=" + (System.currentTimeMillis() - st_epoch) + "\n"); 
		}

		if(PRINT_NETWORK_IS){
			printNetworkWithPass(network, inputs, outputs, "Network after run");
		}

		if(PRINT_NETWORK_TO_FILE_IS){
			String filename = "data/networks/" + NETWORK_RUN_ID + "_post.bal"; 
			
			PrintWriter pw = new PrintWriter(filename); 
			pw.write(network.printNetwork());
			pw.close(); 
			
			if(PRINT_EPOCH_SUMMARY) printBoth("  Network saved to: " + filename + "\n");
		}

		double[] last_measure = null; 
		if(MEASURE_IS) {
			// if max epochs reached, then it also need to be saved (if iStop, then it was saved)  
			last_measure = network.measure(current_epoch, inputs, outputs, !isStop); 
			post_measure.add(last_measure);
		}

		if(HIDDEN_REPRESENTATION_IS){
			hidden_repre_all.add(hidden_repre_cur); 
		}

		if(BAL.MEASURE_IS && BAL.MEASURE_SAVE_AFTER_EACH_RUN){
			network.saveMeasures(NETWORK_RUN_ID, measure_writer);
		}

		if(INPUT_FILEPATH.equals("small.in") || INPUT_FILEPATH.equals("digits.in")){
			printBackwardImages(network, 28, 28); 
		}
		
		// Print out basics 
		double network_result = (last_measure != null) ? last_measure[MEASURE_ERROR] : network.evaluateForward(inputs, outputs);
		
		printBoth("Epochs=" + current_epoch + "\n");
		printBoth("Result=" + network_result + "\n");

		return network; 
	}
	private static double calculateLambda(double init_lambda, int weight_matrix) {
		if(WU_TYPE == WU_BAL_ORIG){
			return  ((weight_matrix == MATRIX_IH || weight_matrix == MATRIX_OH) ? INIT_LAMBDA_V : init_lambda);
		}
		else{
			return init_lambda; 
		}

		//return init_lambda * Math.max(1.0, (100 / (epochs + 50)));

		//return init_lambda * Math.max(0.5, 2 * Math.abs(last_err)); // last error has no reason 
	}

	//interpret activations on the output layer 
	//for example map continuous [0,1] data to discrete {0, 1} 
	public static void postprocessOutput(RealVector out){
		//normal sigmoid 
		//[0.5,\=+\infty] -> 1.0, else 0.0
		
		if(POSTPROCESS_TYPE == POSTPROCESS_SIMPLE){
			for(int i=0; i<out.getDimension() ;i++){
				out.setEntry(i, (out.getEntry(i) >= 0.50) ? 1.0 : 0.0); 
			}
		}

		/*
		//bipolar sigmoid 
		//[0.0,\=+\infty] -> 1.0, else 0.0
		for(int i=0; i<out.getDimension() ;i++){
			out.setEntry(i, (out.getEntry(i) >= 0.00) ? 1.0 : 0.0); 
		}*/

		if(POSTPROCESS_TYPE == POSTPROCESS_MAXIMUM){
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
		}
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
				matrix_data[i][j] = pickFromNormalDistribution(BAL.INIT_NORMAL_DISTRIBUTION_MU, BAL.INIT_SIGMA); 
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
		printBoth("Creating BAL of size ["+in_size + ","+h_size + ","+out_size + "] RunId=" + NETWORK_RUN_ID + "\n");
		//+1 stands for biases 
		this.IH = createInitMatrix(in_size+(isBias(MATRIX_IH)?1:0), h_size);
		this.HO = createInitMatrix(h_size+(isBias(MATRIX_HO)?1:0), out_size);
		this.OH = createInitMatrix(out_size+(isBias(MATRIX_OH)?1:0), h_size);
		this.HI = createInitMatrix(h_size+(isBias(MATRIX_HI)?1:0), in_size);

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
		if(LAMBDA_ERROR_MOMENTUM_IS){
			this.ERR_IH = new double[in_size+1][h_size];
			this.ERR_HO = new double[h_size+1][out_size];
			this.ERR_OH = new double[out_size+1][h_size];
			this.ERR_HI = new double[h_size+1][in_size];
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

		printBoth("Loading matrix [" + cols + "," + rows + "]\n");

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
		printBoth("Creating BAL from file '" + filename + "' RunId=" + NETWORK_RUN_ID + "\n");
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
		printBoth("Bal successfully loaded\n"); 
	}

	//f(net) on a whole layer  
	private void applyNonlinearity(RealVector vector){
		for(int i=0; i<vector.getDimension() ; i++){
			double n = vector.getEntry(i); 
			vector.setEntry(i, 1.0 / (1.0 + Math.exp(-n)));  //normal sigmoid 
			//vector.setEntry(i, 1 - (2 / (1 + Math.exp(-n)))); //bipolar sigmoid 
		}
	}

	private boolean isBias(int which_matrix){
		if(WU_TYPE == WU_GENEREC){
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
	
	private RealVector preMultiply(RealVector rv, int matrix){
		if(matrix == MATRIX_IH) return this.IH.preMultiply(rv);
		if(matrix == MATRIX_HO) return this.HO.preMultiply(rv);
		if(matrix == MATRIX_OH) return this.OH.preMultiply(rv);
		if(matrix == MATRIX_HI) return this.HI.preMultiply(rv);
		return null; 
	}
	
	private static int getMirrorMatrix(int matrix){
		if(matrix == MATRIX_IH) return MATRIX_HI;
		if(matrix == MATRIX_HO) return MATRIX_OH;
		if(matrix == MATRIX_OH) return MATRIX_HO;
		if(matrix == MATRIX_HI) return MATRIX_IH;
		return -1; 
	}
	
	public static RealVector[] swap(RealVector[] rvs){
		RealVector tmp = rvs[0];
		rvs[0] = rvs[2]; 
		rvs[2] = tmp; 
		return rvs; 
	}
	
	private RealVector[] wardPass(RealVector in, int first_matrix, int second_matrix){
		if(INIT_RECIRCULATION_IS){
			return wardPassWithRecirculation(in, first_matrix, second_matrix); 
		}

		RealVector[] ward = new RealVector[3]; 
		ward[0] = addBias(in, first_matrix);
		ward[1] = preMultiply(ward[0], first_matrix);
		applyNonlinearity(ward[1]);
		applyDropoutInPass(ward[1]);
		
		ward[1] = addBias(ward[1], second_matrix);
		ward[2] = preMultiply(ward[1], second_matrix);
		applyNonlinearity(ward[2]);

		return ward; 
	}
	
	//forward activations 
	private RealVector[] forwardPass(RealVector in){
		return wardPass(in, MATRIX_IH, MATRIX_HO); 
	}

	//backward activations 
	private RealVector[] backwardPass(RealVector out){
		return swap(wardPass(out, MATRIX_OH, MATRIX_HI));
	}
	
	private RealMatrix getMatrix(int matrix){
		if(matrix == MATRIX_IH) return this.IH;
		if(matrix == MATRIX_HO) return this.HO;
		if(matrix == MATRIX_OH) return this.OH;
		if(matrix == MATRIX_HI) return this.HI;
		return null; 
	}

	public static RealVector getAverage(RealVector rv1, RealVector rv2){
		return rv1.add(rv2).mapDivide(2.0);
	}

	//TODO test 
	public static double max_fluctuation = 0.0; 
	public static Set<String> max_fluctuation_run_ids = new HashSet<String>(); 
	private RealVector[] wardPassWithRecirculation(RealVector in, int first_matrix, int second_matrix){
		int backward_matrix = getMirrorMatrix(second_matrix); 
		
		RealVector[] ward = new RealVector[3]; 
		ward[0] = addBias(in, first_matrix);
		ward[2] = new ArrayRealVector(getMatrix(second_matrix).getColumnDimension(), 0.0);

		RealVector hidden_net_from_input = preMultiply(ward[0], first_matrix);
		RealVector last_hid_activation = ward[1]; 
		RealVector last_out_activation = ward[2]; 

		ArrayList<RealVector> h = new ArrayList<RealVector>();

		int cc = 0; 
		double max_change = 0.0; 
		for(cc=0; cc < RECIRCULATION_ITERATIONS_MAX ; cc++){
			last_hid_activation = ward[1]; 
			last_out_activation = ward[2];

			ward[2] = addBias(ward[2], backward_matrix); 

			RealVector hidden_net_from_output = preMultiply(ward[2], backward_matrix);

			ward[1] = hidden_net_from_input.add(hidden_net_from_output);
			applyNonlinearity(ward[1]);
			//applyDropoutInPass(forward[1]);
			ward[1] = addBias(ward[1], second_matrix); 

			ward[2] = preMultiply(ward[1], second_matrix);
			applyNonlinearity(ward[2]);

			h.add(ward[1]);
			h.add(ward[2]); 

			// stop when no bigger change 
			max_change = 0.0; 
			for(int j=0; j<last_out_activation.getDimension() ; j++) {
				max_change = Math.max(max_change, Math.abs(last_out_activation.getEntry(j) - ward[2].getEntry(j)));
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

		if(IS_PRINT){ //|| max_fluctuation > 0.05 && !max_fluctuation_run_ids.contains(NETWORK_RUN_ID)){
			max_fluctuation_run_ids.add(NETWORK_RUN_ID);

			System.out.print(((first_matrix == MATRIX_IH) ? "for" : "back") + "wardPassWithRecirculation : " + printVector(in));
			System.out.println("  RUN_ID: " + NETWORK_RUN_ID);
			System.out.println("  Epoch:  " + NETWORK_EPOCH);
			System.out.println("  Max fluctuation: " + DECIMAL_FORMAT.format(max_fluctuation));
			System.out.println("  Iteration count: " + cc);
			System.out.println("  Recirc epsilon : " + DECIMAL_FORMAT.format(RECIRCULATION_EPSILON));
			for(int i=2*RECIRCULATION_ITERATIONS_MAX-10; i<h.size() ; i += 2){System.out.print("  Hidden activations: " + printVector(h.get(i)));}
			for(int i=2*RECIRCULATION_ITERATIONS_MAX-10+1; i<h.size() ; i += 2){System.out.print("  Output activations: " + printVector(h.get(i)));}
			System.out.println("Network: " + this.printNetwork());
			System.out.println();

		}

		if(cc == RECIRCULATION_ITERATIONS_MAX && RECIRCULATION_USE_AVERAGE_WHEN_OSCILATING){
			ward[1] = getAverage(last_hid_activation, ward[1]); 
			ward[2] = getAverage(last_out_activation, ward[2]); 
		}

		return ward; 
	}

	private RealVector[] forwardPassWithRecirculation(RealVector in){
		return wardPassWithRecirculation(in, MATRIX_IH, MATRIX_HO); 
	}

	private RealVector[] backwardPassWithRecirculation(RealVector out){
		return swap(wardPassWithRecirculation(out, MATRIX_OH, MATRIX_HI)); 
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
	//!NOTE: crucial method for performance 
	private static void subLearn(RealMatrix w, RealVector a_pre, RealVector a_post_other, RealVector a_post_self, double lambda, double[][] mom, double[][] batch, double [][] err, boolean[] is_train_pre, boolean[] is_train_post){
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
				if(LAMBDA_ERROR_MOMENTUM_IS){
					err[i][j] = a_post_other.getEntry(j) - a_post_self.getEntry(j); 
				}
			}
		}
	}

	private static void subCHLLearn(RealMatrix w, RealVector a_minus_i, RealVector a_minus_j, RealVector a_plus_i, RealVector a_plus_j, double lambda){
		/*
		System.out.println("subCHLLearn: ");
		System.out.print("  matrix:  " + printMatrix(w));
		System.out.print("  a_plus_i:" + printVector(a_plus_i));
		System.out.print("  a_plus_j:" + printVector(a_plus_j));
		System.out.print("  a_minus_i:" + printVector(a_minus_i));
		System.out.print("  a_minus_j:" + printVector(a_minus_j));
		 */

		int RULE = 0; 
		if(WU_TYPE == WU_BAL_SYM || WU_TYPE == WU_GENEREC_SYM){
			RULE = 1; 
		} 
		if(WU_TYPE == WU_BAL_MID || WU_TYPE == WU_GENEREC_MID){
			RULE = 2; 
		}

		for(int i = 0 ; i < w.getRowDimension() ; i++){
			for(int j = 0 ; j < w.getColumnDimension() ; j++){
				//System.out.println("  " + i + "," + j);
				double w_value = w.getEntry(i, j);

				double dw = 0.0; 
				double aim = a_minus_i.getEntry(i); 
				double aip = a_plus_i.getEntry(i); 
				double ajm = a_minus_j.getEntry(j); 
				double ajp = a_plus_j.getEntry(j);

				if(RULE == 0) dw = lambda * ((aip * ajp) - (aim * ajm));
				if(RULE == 1) dw = (ajp*aim + ajm*aip) - 2*ajm*aim; 
				if(RULE == 2) dw = lambda * (1/2) * (aim + aip) * (ajp - ajm); 
				//System.out.println("   d(" + i + "," + j + "): " + dw);

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
	public void learn(RealVector in, RealVector target){
		if(INIT_TRAIN_ONLY_ON_ERROR){
			if(evaluateForward(in, target) == 0.0){
				printBoth("skip learning on " + printVector(in) + "\n");
				return;
			}
		}

		if(DROPOUT_IS){
			for(int i=0; i<this.active_hidden.length ; i++){
				this.active_hidden[i] = BAL.random.nextBoolean(); 
			}
		}
		
		// shortcuts for dropout 
		boolean[] d_all = BAL.all_true_active_hidden;
		boolean[] d_hidden = this.active_hidden; 

		//System.out.println("Error matrix: " + printMatrix(MatrixUtils.createRealMatrix(ERR_HO)));

		//learn
		if(WU_TYPE == WU_BAL_ORIG){
			//IS_PRINT = true; 
			RealVector[] forward = this.forwardPass(in);
			RealVector[] backward = this.backwardPass(target);
			//IS_PRINT = false; 

			subLearn(this.IH, forward[0], backward[1], forward[1], calculateLambda(INIT_LAMBDA, MATRIX_IH), this.MOM_IH, this.BATCH_IH, this.ERR_IH, d_all, d_hidden); 
			subLearn(this.HO, forward[1], backward[2], forward[2], calculateLambda(INIT_LAMBDA, MATRIX_HO), this.MOM_HO, this.BATCH_HO, this.ERR_HO, d_hidden, d_all); 
			subLearn(this.OH, backward[2], forward[1], backward[1], calculateLambda(INIT_LAMBDA, MATRIX_OH), this.MOM_OH, this.BATCH_OH, this.ERR_OH, d_all, d_hidden); 
			subLearn(this.HI, backward[1], forward[0], backward[0], calculateLambda(INIT_LAMBDA, MATRIX_HI), this.MOM_HI, this.BATCH_HI, this.ERR_HI, d_hidden, d_all);
		}
		if(WU_TYPE == WU_BAL_RECIRC){
			/**/ 
			//IS_PRINT = true; 
			RealVector[] forward = this.forwardPassWithRecirculation(in);
			RealVector[] backward = this.backwardPassWithRecirculation(target);
			//IS_PRINT = false; 

			subLearn(this.IH, forward[0], backward[1], forward[1], calculateLambda(INIT_LAMBDA, MATRIX_IH), this.MOM_IH, this.BATCH_IH, this.ERR_IH, d_all, d_hidden); 
			subLearn(this.HO, forward[1], backward[2], forward[2], calculateLambda(INIT_LAMBDA, MATRIX_HO), this.MOM_HO, this.BATCH_HO, this.ERR_HO, d_hidden, d_all); 

			if(INIT_SYMMETRIC_IS){
				makeSymmetric(this.HI, this.IH, this.IH.getColumnDimension(), this.IH.getRowDimension() - (isBias(MATRIX_IH)?1:0));
				makeSymmetric(this.OH, this.HO, this.HO.getColumnDimension(), this.HO.getRowDimension() - (isBias(MATRIX_HO)?1:0));
			}

			subLearn(this.OH, backward[2], forward[1], backward[1], calculateLambda(INIT_LAMBDA, MATRIX_OH), this.MOM_OH, this.BATCH_OH, this.ERR_OH, d_all, d_hidden); 
			subLearn(this.HI, backward[1], forward[0], backward[0], calculateLambda(INIT_LAMBDA, MATRIX_HI), this.MOM_HI, this.BATCH_HI, this.ERR_HI, d_hidden, d_all);

			if(INIT_SYMMETRIC_IS){
				makeSymmetric(this.IH, this.HI, this.HI.getColumnDimension(), this.HI.getRowDimension() - (isBias(MATRIX_HI)?1:0));
				makeSymmetric(this.HO, this.OH, this.OH.getColumnDimension(), this.OH.getRowDimension() - (isBias(MATRIX_OH)?1:0));
			}
		}
		if(WU_TYPE == WU_GENEREC) {
			//symmetric version 
			//IS_PRINT = true; 
			RealVector[] forward = this.forwardPassWithRecirculation(in); 
			RealVector bothward = this.bothwardPass(in, target); 
			//RealVector biased_target = addBias(target); 
			//IS_PRINT = false; 

			subLearn(this.IH, forward[0], bothward, forward[1], calculateLambda(INIT_LAMBDA, MATRIX_IH), this.MOM_IH, this.BATCH_IH, this.ERR_IH, d_all, d_hidden); 
			subLearn(this.HO, forward[1], target, forward[2], calculateLambda(INIT_LAMBDA, MATRIX_HO), this.MOM_HO, this.BATCH_HO, this.ERR_HO, d_hidden, d_all); 
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
		if(WU_TYPE == WU_GENEREC_CHL || 
				WU_TYPE == WU_GENEREC_MID || 
				WU_TYPE == WU_GENEREC_SYM){ 
			RealVector[] forward = this.forwardPassWithRecirculation(in); 
			RealVector bothward = this.bothwardPass(in, target); 

			subCHLLearn(this.IH, forward[0], forward[1], forward[0], bothward, calculateLambda(INIT_LAMBDA, MATRIX_IH));
			subCHLLearn(this.HO, forward[1], forward[2], addBias(bothward, MATRIX_HO), target, calculateLambda(INIT_LAMBDA, MATRIX_HO));

			makeSymmetric(this.OH, this.HO, this.HO.getColumnDimension(), this.HO.getRowDimension() - (isBias(MATRIX_HO)?1:0));
		}
		if(WU_TYPE == WU_BAL_CHL || 
				WU_TYPE == WU_BAL_MID || 
				WU_TYPE == WU_BAL_SYM){

			//IS_PRINT = true;
			RealVector[] forward = this.forwardPass(in);
			RealVector[] backward = this.backwardPass(target);
			//IS_PRINT = false;

			//RealVector a_minus_i, RealVector a_minus_j, RealVector a_plus_i, RealVector a_plus_j
			subCHLLearn(this.IH, forward[0], forward[1], addBias(backward[0], MATRIX_IH), backward[1], calculateLambda(INIT_LAMBDA, MATRIX_IH)); 
			subCHLLearn(this.HO, forward[1], forward[2], addBias(backward[1], MATRIX_HO), backward[2], calculateLambda(INIT_LAMBDA, MATRIX_HO)); 
			subCHLLearn(this.OH, backward[2], backward[1], addBias(forward[2], MATRIX_OH), forward[1], calculateLambda(INIT_LAMBDA, MATRIX_OH)); 
			subCHLLearn(this.HI, backward[1], backward[0], addBias(forward[1], MATRIX_HI), forward[0], calculateLambda(INIT_LAMBDA, MATRIX_HI)); 
		}

		//log.print(BAL.printVector(forward[1]));
		//log.println(BAL.printVector(backward[1]));
	}

	public double error(RealVector given_activation, RealVector target){
		double error = 0.0; 
		for(int i=0; i<target.getDimension() ; i++){
			error += (given_activation.getEntry(i) - target.getEntry(i)) * (given_activation.getEntry(i) - target.getEntry(i)); 
		}

		return error;  
	}

	
	//evaluates performance on one input-output mapping 
	//returns error 
	public double evaluateForward(RealVector in, RealVector target){
		RealVector[] forward = forwardPass(in);
		if(POSTPROCESS_OUTPUT) postprocessOutput(forward[2]);
		return this.error(forward[forward.length - 1], target); 
	}

	//evaluates performance on several input-output mapping 
	//returns absolute error 
	public double evaluateForward(RealMatrix in, RealMatrix target){
		if(PRINT_EPOCH_SUMMARY) printBoth(" Evaluating forward"); 
		double error = 0.0; 
		for(int i=0; i<in.getRowDimension() ; i++){
			error += this.evaluateForward(in.getRowVector(i), target.getRowVector(i)); 
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

	/** Fetches angle relative to screen center point
	 * where 3 O'Clock is 0 and 12 O'Clock is 270 degrees
	 * 
	 * @param screenPoint
	 * @return angle in degrees from 0-360.
	 */
	public static double getAngle(double dx, double dy)
	{
		double inRads = Math.atan2(dy,dx);

		// We need to map to coordinate system when 0 degree is at 3 O'clock, 270 at 12 O'clock
		if (inRads < 0)
			inRads = Math.abs(inRads);
		else
			inRads = 2*Math.PI - inRads;

		return Math.toDegrees(inRads);
	}

	public void addMeasure(double[] stats){
		for(int i=0; i<MEASURE_COUNT; i++){
			this.measures[i].add(stats[i]); 
		}
	}
	
	//collect monitoring=measure data, epoch is used as identifier
	//  !this data is also stored into measures array 
	public double[] measure(int epoch, RealMatrix in, RealMatrix target, boolean isSave){
		if(PRINT_EPOCH_SUMMARY) printBoth("  measure start [" + in.getRowDimension() + "," + in.getColumnDimension() + "]\n"); 
		
		long start_time = System.currentTimeMillis(); 
		
		double n = in.getRowDimension(); 
		double[] result = new double[MEASURE_COUNT]; 

		if(MEASURE_EPOCH < MEASURE_COUNT) result[MEASURE_EPOCH] = ((double)epoch); 
		if(MEASURE_SIGMA < MEASURE_COUNT) result[MEASURE_SIGMA] = BAL.INIT_SIGMA; 
		if(MEASURE_LAMBDA < MEASURE_COUNT) result[MEASURE_LAMBDA] = BAL.INIT_LAMBDA; 
		if(MEASURE_MOMENTUM < MEASURE_COUNT) result[MEASURE_MOMENTUM] = BAL.INIT_MOMENTUM; 

		if(MEASURE_HIDDEN_FOR_BACK_DIST < MEASURE_COUNT 
				|| MEASURE_ERROR < MEASURE_COUNT
				|| MEASURE_OUTPUT_FOR_BACK_DIST < MEASURE_COUNT 
				|| MEASURE_HIDDEN_DIST < MEASURE_COUNT 
				|| MEASURE_BITSUCC_FORWARD < MEASURE_COUNT
				|| MEASURE_BITSUCC_BACKWARD < MEASURE_COUNT
				|| MEASURE_PATSUCC_FORWARD < MEASURE_COUNT
				|| MEASURE_PATSUCC_BACKWARD < MEASURE_COUNT){
			ArrayList<RealVector> forward_hiddens = new ArrayList<RealVector>(); 
			double hidden_dist = 0.0;
			double hidden_for_back_dist = 0.0;
			double output_for_back_dist = 0.0;

			double bitsucc_f = 0.0; 
			double bitsucc_b = 0.0; 
			double patsucc_f = 0.0; 
			double patsucc_b = 0.0;
			
			double error = 0.0; // archaic measure = bitsucc_f * inputs.size() 

			//double first_second_sum = 0.0; 
			for(int i=0; i<in.getRowDimension(); i++){
				if(PRINT_EPOCH_SUMMARY && (i+1)%1000 == 0) printBoth("    " + (i+1) + " time=" + (System.currentTimeMillis() - start_time) + "\n");
				
				RealVector[] forward = this.forwardPass(in.getRowVector(i));
				RealVector[] backward = this.backwardPass(target.getRowVector(i));

				if(forward[1].getDimension() == backward[1].getDimension()){
					hidden_for_back_dist += forward[1].getDistance(backward[1]) / n;
				}
				if(forward[2].getDimension() == backward[0].getDimension()){
					output_for_back_dist += forward[2].getDistance(backward[0]) / n;
				}

				forward_hiddens.add(forward[1]);

				/*
				double[] output_arr = forward[2].toArray();
				if(output_arr.length > 1){
					Arrays.sort(output_arr); 
					first_second_sum += output_arr[output_arr.length-1] / output_arr[output_arr.length-2];
				}*/

				if(POSTPROCESS_INPUT) postprocessOutput(backward[0]);
				if(POSTPROCESS_OUTPUT) postprocessOutput(forward[2]);
				double err_f = this.error(forward[2], target.getRowVector(i));
				double err_b = this.error(backward[0], in.getRowVector(i)); 
				bitsucc_f += (target.getRowVector(i).getDimension() - err_f) / ((double)(target.getRowVector(i).getDimension())); 
				bitsucc_b += (in.getRowVector(i).getDimension() - err_b) / ((double)(in.getRowVector(i).getDimension()));  
				patsucc_f += (err_f <= 0.0) ? 1.0 : 0.0; 
				patsucc_b += (err_b <= 0.0) ? 1.0 : 0.0;
				error += err_f; 
			}
			if(MEASURE_HIDDEN_FOR_BACK_DIST < MEASURE_COUNT) result[MEASURE_HIDDEN_FOR_BACK_DIST] = hidden_for_back_dist; 
			if(MEASURE_OUTPUT_FOR_BACK_DIST < MEASURE_COUNT) result[MEASURE_OUTPUT_FOR_BACK_DIST] = output_for_back_dist; 

			if(MEASURE_ERROR < MEASURE_COUNT) result[MEASURE_ERROR] = error;
			if(MEASURE_BITSUCC_FORWARD < MEASURE_COUNT) result[MEASURE_BITSUCC_FORWARD] = bitsucc_f / ((double)(target.getRowDimension()));
			if(MEASURE_BITSUCC_BACKWARD < MEASURE_COUNT) result[MEASURE_BITSUCC_BACKWARD] = bitsucc_b / ((double)(in.getRowDimension()));
			if(MEASURE_PATSUCC_FORWARD < MEASURE_COUNT) result[MEASURE_PATSUCC_FORWARD] = patsucc_f / ((double)(target.getRowDimension()));
			if(MEASURE_PATSUCC_BACKWARD < MEASURE_COUNT) result[MEASURE_PATSUCC_BACKWARD] = patsucc_b / ((double)(in.getRowDimension()));

			if(MEASURE_HIDDEN_DIST < MEASURE_COUNT){
				long st = System.currentTimeMillis(); 
				
				for(int i=0; i<forward_hiddens.size() ; i++){
					for(int j=i+1; j<forward_hiddens.size() ; j++){
						hidden_dist += forward_hiddens.get(i).getDistance(forward_hiddens.get(j)) / (forward_hiddens.size() * (forward_hiddens.size() + 1) / 2); 
					}
				}

				result[MEASURE_HIDDEN_DIST] = hidden_dist;
				if(PRINT_EPOCH_SUMMARY) printBoth("    hidden_dist_time=" + (System.currentTimeMillis() - st) + "\n"); 
			}

			if(MEASURE_IN_TRIANGLE < MEASURE_COUNT){
				if(forward_hiddens.get(0).getDimension() == 2){
					ArrayList<Point> hidden_points = new ArrayList<Point>(); 
					for(int i=0; i<forward_hiddens.size() ; i++){
						hidden_points.add(new Point((int)(1000.0 * forward_hiddens.get(i).getEntry(0)), (int)(1000.0 * forward_hiddens.get(i).getEntry(1)))); 
					}
					ArrayList<Point> convex_hull = ConvexHull.execute(hidden_points); 
					result[MEASURE_IN_TRIANGLE] = (double)(hidden_points.size() - convex_hull.size());
	
					/*//DEVELOPER DEBUG 
					log.println("Hidden points");
					log.println(hidden_points);
					log.println("ConvexHull points");
					log.println(convex_hull);
					log.println("End"); */
				}
				else{
					result[MEASURE_IN_TRIANGLE] = -1.0; 
				}
			}

		}

		if(MEASURE_FLUCTUATION < MEASURE_COUNT){
			long st = System.currentTimeMillis(); 
			max_fluctuation = 0.0; 
			for(int i=0; i<in.getRowDimension(); i++){
				this.forwardPassWithRecirculation(in.getRowVector(i));
				this.backwardPassWithRecirculation(target.getRowVector(i));
			}
			result[MEASURE_FLUCTUATION] = max_fluctuation; 
			max_fluctuation = 0.0; 
			if(PRINT_EPOCH_SUMMARY) printBoth("    fluctuation_time=" + (System.currentTimeMillis() - st) + "\n"); 
		}

		if(MEASURE_MATRIX_AVG_W < MEASURE_COUNT){
			double matrix_avg_w = 0.0;
			matrix_avg_w = (sumAbsoluteValuesOfMatrixEntries(this.IH) + sumAbsoluteValuesOfMatrixEntries(this.HO) + sumAbsoluteValuesOfMatrixEntries(this.OH) + sumAbsoluteValuesOfMatrixEntries(this.IH)) / (this.IH.getColumnDimension()*this.IH.getRowDimension() + this.HO.getColumnDimension()*this.HO.getRowDimension()+ this.OH.getColumnDimension()*this.OH.getRowDimension()+ this.HI.getColumnDimension()*this.HI.getRowDimension()); 
			result[MEASURE_MATRIX_AVG_W] = matrix_avg_w;
		}

		if(MEASURE_MATRIX_SIMILARITY < MEASURE_COUNT){
			double matrix_similarity = 0.0;
			if(MEASURE_MATRIX_SIMILARITY >= 0 && this.HO.getColumnDimension() == this.HI.getColumnDimension() && this.HO.getRowDimension() == this.HI.getRowDimension()
					&& this.OH.getColumnDimension() == this.IH.getColumnDimension() && this.OH.getRowDimension() == this.IH.getRowDimension()){
				RealMatrix diff_HO_HI = this.HO.subtract(this.HI); 
				RealMatrix diff_OH_IH = this.OH.subtract(this.IH);
				matrix_similarity = (sumAbsoluteValuesOfMatrixEntries(diff_HO_HI) + sumAbsoluteValuesOfMatrixEntries(diff_OH_IH)) / (this.IH.getColumnDimension()*this.IH.getRowDimension() + this.HI.getColumnDimension()*this.HI.getRowDimension());
			}   
			result[MEASURE_MATRIX_SIMILARITY] = matrix_similarity;
		}

		if(MEASURE_LAMBDA_V < MEASURE_COUNT){
			result[MEASURE_LAMBDA_V] = BAL.INIT_LAMBDA_V; 
		}
		
		if(isSave){
			addMeasure(result); 
		}
		
		MEASURE_EXECUTION_TIME += System.currentTimeMillis() - start_time; 
		if(PRINT_EPOCH_SUMMARY) printBoth("  measure time=" + (System.currentTimeMillis() - start_time) + "\n"); 
		
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

				s_child += DECIMAL_FORMAT.format(val) + "\t";
				if(id != BAL.MEASURE_GROUP_BY){
					s_parent += DECIMAL_FORMAT.format(val) + "\t"; 
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
			if(j != 0) sb.append("\t");
			sb.append(BAL.MEASURE_HEADINGS[BAL.MEASURE_GROUP_BY_COLS[j]]);
		}
		sb.append(" success sample_ratio\n");

		List<String> result = new ArrayList<String>(); 
		for(Entry<String, Integer> entry : counts_child.entrySet()){
			Integer child_count = entry.getValue();
			Integer parent_count = counts_parent.get(child2parent.get(entry.getKey())); 
			result.add(entry.getKey() + DECIMAL_FORMAT.format(100.0*((double)child_count / (double)parent_count)) + " " + child_count + "/" + parent_count);
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
			log.print("avg("+BAL.MEASURE_HEADINGS[j]+")=" + DECIMAL_FORMAT.format(sum[j] / ((double)measures.size())) + "\n");
		}
	}

	public static RealMatrix loadFromFile(String filepath){
		printBoth("Loading matrix from file " + filepath + "\n");
		long st = System.currentTimeMillis(); 
		
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

		printBoth("Loading matrix from file DONE in time=" + (System.currentTimeMillis() - st) + "\n");
		return MatrixUtils.createRealMatrix(data);
	}

	public static String printVector(RealVector v){
		StringBuilder sb = new StringBuilder();

		for(int i = 0 ; i < v.getDimension() ; i++){
			if(i!=0){
				sb.append(' ');
			}
			sb.append(DECIMAL_FORMAT.format(v.getEntry(i)));
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
				sb.append(DECIMAL_FORMAT.format(m.getEntry(i, j)));
			}
			sb.append('\n');
		}
		return sb.toString(); 
	}

	public static String printTwoDimArray(double[][] m){
		StringBuilder sb = new StringBuilder();
		sb.append(m.length);
		sb.append(' '); 
		sb.append(m[0].length);
		sb.append('\n'); 

		for(int i = 0 ; i < m.length ; i++){
			for(int j = 0 ; j < m[0].length ; j++){
				if(j!=0){
					sb.append(' ');
				}
				sb.append(DECIMAL_FORMAT.format(m[i][j]));
			}
			sb.append('\n');
		}
		return sb.toString(); 
	}

	public static void printForwardPass(RealVector[] forward, RealVector target){
		printBoth("Forward pass:");

		for(int j=0; j<forward.length; j++){
			log.print(BAL.printVector(forward[j]));
			System.out.print(BAL.printVector(forward[j]));
		}

		BAL.postprocessOutput(forward[2]);

		printBoth("Given:   " + BAL.printVector(forward[2]));
		printBoth("Expected:" + BAL.printVector(target));
	}

	public static void printBackwardPass(RealVector[] backward){
		log.println("Backward pass:");
		System.out.println("Backward pass:");

		for(int j=0; j<3; j++){
			log.print(BAL.printVector(backward[j]));
			System.out.print(BAL.printVector(backward[j]));
		}
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

	public String printMomentum(){
		StringBuilder sb = new StringBuilder(); 
		sb.append((this.IH.getRowDimension()-1) + " " + this.IH.getColumnDimension() + " " + this.HO.getColumnDimension() + "\n");
		sb.append("#MOM_IH\n"); 
		sb.append(BAL.printTwoDimArray(this.MOM_IH));
		sb.append("#MOM_HO\n"); 
		sb.append(BAL.printTwoDimArray(this.MOM_HO));
		sb.append("#MOM_OH\n"); 
		sb.append(BAL.printTwoDimArray(this.MOM_OH));
		sb.append("#MOM_HI\n"); 
		sb.append(BAL.printTwoDimArray(this.MOM_HI));
		return sb.toString(); 
	}


	public static void printNetworkWithPass(BAL network, RealMatrix inputs, RealMatrix outputs, String title){
		log.println("---------- " + title + " --------------");
		log.println(network.printNetwork());

		System.out.println("----------" + title + "--------------");
		System.out.println(network.printNetwork());

		for(int i=0 ; i<inputs.getRowDimension(); i++){
			RealVector[] forward = network.forwardPass(inputs.getRowVector(i));
			printForwardPass(forward, outputs.getRowVector(i));
		}
		for(int i=0 ; i<outputs.getRowDimension(); i++){
			RealVector[] backward = network.backwardPass(outputs.getRowVector(i));
			printBackwardPass(backward);
		}
	}

	//TODO Refactor: merge with printPreAndPostMeasures()
	//saves this.measures into file 
	public boolean saveMeasures(String run_id, PrintWriter writer){
		if(!MEASURE_IS) return true; 

		writer.write("RUN_ID");
		for(int i=0; i<MEASURE_COUNT ; i++){
			writer.write("\t");
			writer.write(MEASURE_HEADINGS[i]); 
		}
		writer.println(); 

		for(int i=0; i<this.measures[0].size(); i++){
			writer.write(run_id);
			for(int j=0; j<this.measures.length; j++){
				writer.write("\t"); 
				writer.print(this.measures[j].get(i)); 
			}
			writer.println(); 
		}

		return true; 
	}	

	//TODO Refactor with saveMeasures()
	public static void printMeasure(String global_run_id, String sufix, ArrayList<double[]> m) throws FileNotFoundException, UnsupportedEncodingException{
		if(!MEASURE_IS) return; 

		PrintWriter writer = new PrintWriter("data/" + global_run_id + "_" + sufix + ".csv", "UTF-8");

		for(int i=0; i<MEASURE_COUNT ; i++){
			if(i != 0){ writer.write('\t'); }
			
			writer.write(MEASURE_HEADINGS[i]); 
		}
		writer.println();

		for(int i=0; i<m.size() ; i++){
			for(int j=0; j<MEASURE_COUNT ; j++){
				if(j != 0){ writer.write('\t'); }
				
				writer.print(DECIMAL_FORMAT.format(m.get(i)[j])); 
			}
			writer.println();
		}

		writer.close();

		log.print("Measure_" + sufix + " : Averages");
		BAL.measureAverages(m);
		
		printBoth("Measure_" + sufix + ": GroupBy\n");
		BAL.measureGroupBY(m); 
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
	
	public static void printBackwardImages(BAL network, int width, int height) throws IOException{ 
		RealMatrix patterns = MatrixUtils.createRealIdentityMatrix(10); 
		
		//TODO wrong rotation 
		for(int pi = 0; pi < patterns.getRowDimension(); pi++){
			RealVector[] backward = network.backwardPass(patterns.getRowVector(pi));
			BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			
			for(int w=0, d=0; w < width; w++){
				for(int h=0; h < height ; h++, d++){
					int gray = (int)(backward[0].getEntry(d) * 255.0);
					img.setRGB(w, h, (gray << 16) + (gray << 8) + gray);
				}
			}
			
			ImageIO.write(img, "bmp", new File("data/img/" + NETWORK_RUN_ID + "_" + pi + ".bmp"));
		}
	}

	public static void generateNetworkRunId(){
		NETWORK_RUN_ID = INPUT_FILEPATH.substring(0, INPUT_FILEPATH.indexOf('.')) + "_" + WU_TYPE + "_" + INIT_HIDDEN_LAYER_SIZE + "_" + (System.currentTimeMillis());
	}

	public static String experimentInit() throws IOException{
		generateNetworkRunId(); 
		String global_run_id = NETWORK_RUN_ID; // it changes after each new network generation 

		log = new PrintWriter("data/" + global_run_id + ".log");

		measure_writer = new PrintWriter("data/" + global_run_id + "_measure.csv");
		pre_measure = new ArrayList<double[]>();
		post_measure = new ArrayList<double[]>();
		test_measure = new ArrayList<double[]>();

		if(HIDDEN_REPRESENTATION_IS){
			hidden_repre_all = new ArrayList<ArrayList<RealVector[]>>(); 
		}

		return global_run_id; 
	}

	public static void experimentRun(BAL network) throws IOException {
		RealMatrix inputs = BAL.loadFromFile(BAL.INPUT_FILEPATH);
		RealMatrix outputs = BAL.loadFromFile(BAL.OUTPUT_FILEPATH);
		RealMatrix test_inputs = (TEST_INPUT_FILEPATH == null) ? null : loadFromFile(TEST_INPUT_FILEPATH);
		RealMatrix test_outputs = (TEST_OUTPUT_FILEPATH == null) ? null : loadFromFile(TEST_OUTPUT_FILEPATH);
		
		int ri = 0; 

		while(ri<BAL.INIT_RUNS) {
			for(int si=0; si<BAL.TRY_SIGMA.length; si++) {
				for(int mi=0; mi<BAL.TRY_MOMENTUM.length ; mi++){
					for(int li=0; li<BAL.TRY_LAMBDA.length ; li++){
						for(int tlri=0; tlri<BAL.TRY_LAMBDA_V.length ; tlri++){
							ri++; 
							if(ri > BAL.INIT_RUNS){
								break;
							}

							BAL.INIT_SIGMA = BAL.TRY_SIGMA[si]; 
							BAL.INIT_LAMBDA = BAL.TRY_LAMBDA[li];
							BAL.INIT_MOMENTUM = BAL.TRY_MOMENTUM[mi];
							BAL.INIT_LAMBDA_V = BAL.TRY_LAMBDA_V[tlri];

							long start_time = System.currentTimeMillis(); 
							MEASURE_EXECUTION_TIME = 0; 

							log.println("  ======== " + ri + "/" + BAL.INIT_RUNS + " ==============");
							System.out.println("  ======== " + ri + "/" + BAL.INIT_RUNS + " ==============");

							BAL N = BAL.run(network, inputs, outputs);

							printBoth("RunTime=" + (System.currentTimeMillis() - start_time) + "\n");
							printBoth("MeasureTime=" + MEASURE_EXECUTION_TIME + "\n"); 
							
							if(test_inputs != null){
								if(PRINT_EPOCH_SUMMARY) printBoth("Testing on " + test_inputs.getRowDimension() + " samples\n");
								double[] m = N.measure(0, test_inputs, test_outputs, false); 
								test_measure.add(m);
							}
						}
					}
				}
			}
		}


		//BAL.INIT_NOISE_SPAN = BAL.TRY_NOISE_SPAN[random.nextInt(BAL.TRY_NOISE_SPAN.length)];
		//BAL.INIT_MULTIPLY_WEIGHTS = BAL.TRY_MULTIPLY_WEIGHTS[random.nextInt(BAL.TRY_MULTIPLY_WEIGHTS.length)];
	}

	public static void experimentFinalize(String global_run_id) throws FileNotFoundException, UnsupportedEncodingException {
		printMeasure(global_run_id, "pre", pre_measure);
		printMeasure(global_run_id, "post", post_measure);
		
		if(TEST_INPUT_FILEPATH != null) printMeasure(global_run_id, "test", test_measure);
		
		printHiddenRepresentations(); 
		log.close(); 
		measure_writer.close(); 
	}

	public static void experiment_Default() throws IOException{
		String global_run_id = experimentInit();
		experimentRun(null);
		experimentFinalize(global_run_id);

		/*
		// When recirculation with iterative activation is allowed it prints individual iteration counts  
		if(!recirc_iter_counts.isEmpty()){
			int sum=0; 
			for(int i=0; i<recirc_iter_counts.size(); i++) {
				System.out.println(recirc_iter_counts.get(i));
				sum += recirc_iter_counts.get(i); 
			}
			System.out.println("Iteration recirc avg=" + (sum / recirc_iter_counts.size()) + " sum=" + sum + " count=" + recirc_iter_counts.size());
		} */
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
		MEASURE_SAVE_AFTER_EACH_RUN = true; 
		MEASURE_RECORD_EACH = 10000;

		INPUT_FILEPATH = input_prefix + ".in"; 
		OUTPUT_FILEPATH = input_prefix + ".out"; 
		POSTPROCESS_INPUT = true; 
		POSTPROCESS_OUTPUT = true; 

		INIT_LAMBDA = 0.7; 
		INIT_MAX_EPOCHS = 1000;
		INIT_RUNS = 20000; 
		INIT_CANDIDATES_COUNT = 0;
		INIT_SHUFFLE_IS = true;
		INIT_BATCH_IS = false;
		INIT_TRAIN_ONLY_ON_ERROR = false; 

		LAMBDA_ERROR_MOMENTUM_IS = false; 
		INIT_LAMBDA_V = 0.001;   

		RECIRCULATION_EPSILON = 0.001; //if the max unit activation change is less the RECIRCULATION_EPSILON, it will stop 
		RECIRCULATION_ITERATIONS_MAX = 200; //maximum number of iterations to approximate the underlying dynamic system  
		RECIRCULATION_USE_AVERAGE_WHEN_OSCILATING = true;

		DROPOUT_IS = false; 
		CONVERGENCE_NO_CHANGE_FOR = -1; 
		STOP_IF_NO_ERROR = false;
		STOP_IF_NO_IMPROVE_FOR = -1; 

		INIT_MOMENTUM_IS = true;
		INIT_MOMENTUM = 0.0;  

		INIT_NORMAL_DISTRIBUTION_MU = 0;
		NORMAL_DISTRIBUTION_SPAN = 15; 

		HIDDEN_REPRESENTATION_IS = false;
		HIDDEN_REPRESENTATION_DIRECTORY = "data/" + input_prefix; 
		HIDDEN_REPRESENTATION_EACH = 1; 
		HIDDEN_REPRESENTATION_AFTER = 200;
		HIDDEN_REPRESENTATION_ONLY_EACH = 200;

		PRINT_NETWORK_IS = false;  
		PRINT_NETWORK_TO_FILE_IS = false;

		TRY_LAMBDA = new double[]{0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0};

		TRY_LAMBDA_V = new double[]{0.0000001, 0.0000002, 0.0000005, 0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 
				0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 
				0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 
				100.0};

		//for(int h=3; h<=144; h += h/8 + 1){
		for(int h=4; h<=20 ; h++){
			INIT_HIDDEN_LAYER_SIZE = h; 
			experiment_Default();
		} 
	}

	public static void experiment_TestImplementation() throws IOException{
		MEASURE_IS = true; 
		MEASURE_SAVE_AFTER_EACH_RUN = true; 
		MEASURE_RECORD_EACH = 100000;

		INPUT_FILEPATH = "auto4.in"; 
		OUTPUT_FILEPATH = "auto4.in"; 
		INIT_HIDDEN_LAYER_SIZE = 2;
		POSTPROCESS_INPUT = true; 
		POSTPROCESS_OUTPUT = true; 
		POSTPROCESS_TYPE = POSTPROCESS_SIMPLE; 
		
		INIT_CANDIDATES_COUNT = 0;
		INIT_SHUFFLE_IS = true;
		INIT_BATCH_IS = false;
		INIT_TRAIN_ONLY_ON_ERROR = false; 

		LAMBDA_ERROR_MOMENTUM_IS = false; 

		//TRY_LAMBDA = new double[]{0.1, 0.7, 3.0}; 
		/*TRY_LAMBDA = new double[]{
				0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 
				0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 
				50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 
				100000.0, 200000.0, 500000.0, 1000000.0, 2000000.0, 5000000.0, 10000000.0};*/  
		TRY_LAMBDA = new double[]{
				0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 
				1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 
				100000.0, 300000.0, 1000000.0, 3000000.0, 10000000.0, 30000000.0, 100000000.0, 300000000.0, 1000000000.0
		};
		
		//TRY_LAMBDA_V = new double[]{0.1, 0.7, 3.0}; 
		/*TRY_LAMBDA_V = new double[]{
				0.00000001, 0.00000002, 0.00000005, 0.0000001, 0.0000002, 0.0000005, 0.000001, 0.000002, 0.000005, 0.00001, 
				0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 
				0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 
				100.0};*/  
		TRY_LAMBDA_V = new double[]{
				0.0000000001, 0.0000000003, 0.000000001, 0.000000003, 0.00000001, 0.00000003, 0.0000001, 0.0000003, 0.000001, 0.000003,
				0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 
				1.0, 3.0, 10.0, 30.0, 100.0
		};
		
		TRY_SIGMA = new double[]{2.3};
		//TRY_SIGMA = new double[]{1.0, 2.3, 10.0};
		INIT_MOMENTUM_IS = false;
		TRY_MOMENTUM = new double[]{0.0}; 
		//TRY_MOMENTUM = new double[]{0.0, 0.01, 0.1};

		INIT_MAX_EPOCHS = 10000;
		INIT_RUNS = 100 * TRY_LAMBDA.length * TRY_LAMBDA_V.length * TRY_SIGMA.length * TRY_MOMENTUM.length;

		RECIRCULATION_EPSILON = 0.001; //if the max unit activation change is less the RECIRCULATION_EPSILON, it will stop 
		RECIRCULATION_ITERATIONS_MAX = 50; //maximum number of iterations to approximate the underlying dynamic system  
		RECIRCULATION_USE_AVERAGE_WHEN_OSCILATING = true;

		DROPOUT_IS = false; 
		CONVERGENCE_NO_CHANGE_FOR = -1; 
		STOP_IF_NO_ERROR = true;
		STOP_IF_NO_IMPROVE_FOR = -1; 

		INIT_MOMENTUM_IS = true;
		INIT_MOMENTUM = 0.0;  

		INIT_NORMAL_DISTRIBUTION_MU = 0;
		NORMAL_DISTRIBUTION_SPAN = 15; 

		HIDDEN_REPRESENTATION_IS = false;
		HIDDEN_REPRESENTATION_DIRECTORY = "data/test/"; 
		HIDDEN_REPRESENTATION_EACH = 1; 
		HIDDEN_REPRESENTATION_AFTER = 200;
		HIDDEN_REPRESENTATION_ONLY_EACH = 200;

		PRINT_NETWORK_IS = false;  
		PRINT_NETWORK_TO_FILE_IS = false;

		MEASURE_COUNT = MEASURE_FLUCTUATION;
		experiment_Default();
	}

	public static void experiment_Digits() throws IOException{
		MEASURE_IS = true;
		MEASURE_COUNT = MEASURE_HIDDEN_DIST; 
		MEASURE_SAVE_AFTER_EACH_RUN = true; 
		MEASURE_RECORD_EACH = 1;
		MEASURE_GROUP_BY_COLS = new int[]{MEASURE_PATSUCC_FORWARD, MEASURE_SIGMA, MEASURE_LAMBDA, MEASURE_LAMBDA_V, MEASURE_MOMENTUM};

		//INPUT_FILEPATH = "small.in"; 
		//OUTPUT_FILEPATH = "small.out";
		INPUT_FILEPATH = "digits.in"; 
		OUTPUT_FILEPATH = "digits.out"; 
		TEST_INPUT_FILEPATH = "digits.test.in"; 
		TEST_OUTPUT_FILEPATH = "digits.test.out";
		
		INIT_HIDDEN_LAYER_SIZE = 300;
		POSTPROCESS_INPUT = false; 
		POSTPROCESS_OUTPUT = true; 
		POSTPROCESS_TYPE = POSTPROCESS_MAXIMUM; 

		INIT_CANDIDATES_COUNT = 0;
		INIT_SHUFFLE_IS = true;
		INIT_BATCH_IS = false;
		INIT_TRAIN_ONLY_ON_ERROR = false; 

		LAMBDA_ERROR_MOMENTUM_IS = false;

		TRY_SIGMA = new double[]{1 / (Math.sqrt(784 + 1))};

		INIT_MOMENTUM_IS = true;
		TRY_MOMENTUM = new double[]{0.01}; 
		
		//TRY_LAMBDA = new double[]{1.0};
		//TRY_LAMBDA = new double[]{0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0};
		/* TRY_LAMBDA = new double[]{
				0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 
				50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0
		}; */
		TRY_LAMBDA = new double[]{
				0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 1000000.0
		};
		
		//TRY_LAMBDA_V = new double[]{0.00001};
		//TRY_LAMBDA_V = new double[]{0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0};
		/* TRY_LAMBDA_V = new double[]{
				0.0000001, 0.0000002, 0.0000005, 0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 
				0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05
		}; */
		TRY_LAMBDA_V = new double[]{
				0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01
		};

		INIT_MAX_EPOCHS = 20;
		INIT_RUNS = 1 * TRY_LAMBDA.length * TRY_LAMBDA_V.length * TRY_SIGMA.length * TRY_MOMENTUM.length; 
		STOP_IF_NO_IMPROVE_FOR = 3; 
		
		RECIRCULATION_EPSILON = 0.001; //if the max unit activation change is less the RECIRCULATION_EPSILON, it will stop 
		RECIRCULATION_ITERATIONS_MAX = 200; //maximum number of iterations to approximate the underlying dynamic system  
		RECIRCULATION_USE_AVERAGE_WHEN_OSCILATING = true;

		DROPOUT_IS = false; 
		CONVERGENCE_NO_CHANGE_FOR = -1; 
		STOP_IF_NO_ERROR = false; // in digits that's impossible -> this way we will spare one pass through whole 
		
		INIT_NORMAL_DISTRIBUTION_MU = 0;
		NORMAL_DISTRIBUTION_SPAN = 15; 

		HIDDEN_REPRESENTATION_IS = false;
		HIDDEN_REPRESENTATION_DIRECTORY = "data/test/"; 
		HIDDEN_REPRESENTATION_EACH = 1; 
		HIDDEN_REPRESENTATION_AFTER = 200;
		HIDDEN_REPRESENTATION_ONLY_EACH = 200;

		PRINT_NETWORK_IS = false;  
		PRINT_NETWORK_TO_FILE_IS = false;
		PRINT_EPOCH_SUMMARY = true; 

		experiment_Default();
	}
}
