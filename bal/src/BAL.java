import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class BAL {
	private RealMatrix IH; 
	private RealMatrix HO;  
	private RealMatrix OH; 
	private RealMatrix HI;

	public static final String[] MEASURE_HEADINGS = {"epoch","err","h_dist","h_f_b_dist","m_avg_w","m_sim"}; 
	
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
	
	public static final int MEASURE_COUNT = 6; 
	
	private ArrayList<Double>[] measures = null; 
	
	private double normalDistributionValue(double mu, double sigma, double x){
		return (1 / (Math.sqrt(2*Math.PI*sigma*sigma))) * Math.exp(- ((Math.pow(x - mu, 2))/(2*sigma*sigma)));
	}
	
	private RealMatrix createInitMatrix(int rows, int cols){
		double [][] matrix_data = new double[rows][cols]; 
		for(int i=0; i<rows; i++){
			for(int j=0; j<cols; j++){
				matrix_data[i][j] = normalDistributionValue(0, 1/Math.sqrt(rows + 1), Math.random()*5 - 2.5); 
			}
		}
		
		RealMatrix m = MatrixUtils.createRealMatrix(matrix_data);
		return m; 
	}
	
	public BAL(int in_size, int h_size, int out_size) {
		this.IH = createInitMatrix(in_size, h_size);
		this.HO = createInitMatrix(h_size, out_size);
		this.OH = createInitMatrix(out_size, h_size);
		this.HI = createInitMatrix(h_size, in_size);
		
		this.measures = new ArrayList[MEASURE_COUNT]; 
		for(int i=0; i<MEASURE_COUNT; i++){
			this.measures[i] = new ArrayList<Double>();
		}
	}
	
	//TODO consider k*n in exponent 
	private void applyNonlinearity(RealVector vector){
		for(int i=0; i<vector.getDimension() ; i++){
			double n = vector.getEntry(i); 
			vector.setEntry(i, 1 / (1 + Math.exp(-n)));
		}
	}
	
	private RealVector[] forwardPass(RealVector in){
		RealVector[] forward = new RealVector[3]; 
		forward[0] = in;
		
		forward[1] = this.IH.preMultiply(forward[0]);
		applyNonlinearity(forward[1]); 
		forward[2] = this.HO.preMultiply(forward[1]);
		applyNonlinearity(forward[2]);

		return forward; 
	}
	
	private RealVector[] backwardPass(RealVector out){
		RealVector[] backward = new RealVector[3]; 
		backward[2] = out; 
		
		backward[1] = this.OH.preMultiply(backward[2]);
		applyNonlinearity(backward[1]); 
		backward[0] = this.HI.preMultiply(backward[1]);
		applyNonlinearity(backward[0]);
		
		return backward; 
	}
	
	//\delta w_{pq}^F = \lambda a_p^{F}(a_q^{B} - a_q^{F})
	//\delta w_ij = lamda * a_pre * (a_post_other - a_post_self)  
	private void subLearn(RealMatrix w, RealVector a_pre, RealVector a_post_other, RealVector a_post_self, double lambda){
		for(int i = 0 ; i < w.getRowDimension() ; i++){
			for(int j = 0 ; j < w.getColumnDimension() ; j++){
				double w_value = w.getEntry(i, j); 
				w.setEntry(i, j, w_value + lambda * a_pre.getEntry(i) * (a_post_other.getEntry(j) - a_post_self.getEntry(j)));
			}
		}
	}
	
	public void learn(RealVector in, RealVector target, double lambda){
		RealVector[] forward = this.forwardPass(in);
		RealVector[] backward = this.backwardPass(target);
		
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
		
		//learn 
		subLearn(this.IH, forward[0], backward[1], forward[1], lambda); 
		subLearn(this.HO, forward[1], backward[2], forward[2], lambda); 
		subLearn(this.OH, backward[2], forward[1], backward[1], lambda); 
		subLearn(this.HI, backward[1], forward[0], backward[0], lambda); 
	}
	
	//for example map continuous [0,1] data to discrete {0, 1} 
	protected void postprocessOutput(RealVector out){
		//[0.5,\=+\infty] -> 1.0, else 0.0
		/*
		for(int i=0; i<out.getDimension() ;i++){
			out.setEntry(i, (out.getEntry(i) >= 0.5) ? 1.0 : 0.0); 
		}*/
		
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
	}
	
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
	
	public double evaluate(RealMatrix in, RealMatrix target){
		double error = 0.0; 
		for(int i=0; i<in.getRowDimension() ; i++){
			error += this.evaluate(in.getRowVector(i), target.getRowVector(i)); 
		}
		return error; 
	}
	
	public static double sumMatrixEntries(RealMatrix m){
		double sum = 0.0; 
		for(int i = 0 ; i < m.getRowDimension() ; i++){
			for(int j = 0 ; j < m.getColumnDimension() ; j++){
				sum += Math.abs(m.getEntry(i, j)); 
			}
		}
		return sum; 
	}
	
	public void measure(int epoch, RealMatrix in, RealMatrix target){
		double n = in.getRowDimension(); 
		
		double hidden_dist = 0.0;
		double for_back_dist = 0.0;
		double matrix_avg_w = 0.0;
		double matrix_similarity = 0.0;

		this.measures[MEASURE_EPOCH].add((double)epoch); 
		
		this.measures[MEASURE_ERROR].add(this.evaluate(in, target)); 
		
		ArrayList<RealVector> forward_hiddens = new ArrayList<RealVector>(); 
		
		for(int i=0; i<in.getRowDimension(); i++){
			RealVector[] forward = this.forwardPass(in.getRowVector(i));
			RealVector[] backward = this.backwardPass(target.getRowVector(i));
			
			for_back_dist += forward[1].getDistance(backward[1]) / n; 
			forward_hiddens.add(forward[1]); 
		}
		this.measures[MEASURE_HIDDEN_FOR_BACK_DIST].add(for_back_dist); 

		for(int i=0; i<forward_hiddens.size() ; i++){
			for(int j=i+1; j<forward_hiddens.size() ; j++){
				hidden_dist += forward_hiddens.get(i).getDistance(forward_hiddens.get(j)) / (forward_hiddens.size() * (forward_hiddens.size() + 1) / 2); 
			}
		}
		
		this.measures[MEASURE_HIDDEN_DIST].add(hidden_dist);  
		
		matrix_avg_w = (sumMatrixEntries(this.IH) + sumMatrixEntries(this.HO) + sumMatrixEntries(this.OH) + sumMatrixEntries(this.IH)) / (this.IH.getColumnDimension()*this.IH.getRowDimension() + this.HO.getColumnDimension()*this.HO.getRowDimension()+ this.OH.getColumnDimension()*this.OH.getRowDimension()+ this.HI.getColumnDimension()*this.HI.getRowDimension()); 
		this.measures[MEASURE_MATRIX_AVG_W].add(matrix_avg_w);   
		
		if(MEASURE_MATRIX_SIMILARITY >= 0 && this.HO.getColumnDimension() == this.HI.getColumnDimension() && this.HO.getRowDimension() == this.HI.getRowDimension()){
			RealMatrix diff_HO_HI = this.HO.subtract(this.HI); 
			RealMatrix diff_OH_IH = this.OH.subtract(this.IH);
			matrix_similarity = (sumMatrixEntries(diff_HO_HI) + sumMatrixEntries(diff_OH_IH)) / (this.IH.getColumnDimension()*this.IH.getRowDimension() + this.HI.getColumnDimension()*this.HI.getRowDimension());   
			this.measures[MEASURE_MATRIX_SIMILARITY].add(matrix_similarity);
		}
		
	}
	
	public boolean saveMeasure(String filename){
		PrintWriter writer;
		try {
			writer = new PrintWriter(filename, "UTF-8");
		} catch (Exception e) {
			return false; 
		} 
		
		double[] m = new double[MEASURE_COUNT];
		for(int j=0; j<MEASURE_COUNT; j++){
			m[j] = 0;
		}
		
		for(int i=0; i<this.measures[0].size(); i++){
			for(int j=0; j<this.measures.length; j++){
				m[j] = Math.max(m[j], this.measures[j].get(i));
			}
		}
		m[MEASURE_EPOCH] = 1; 
		
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
		sb.append("BAL network of size " + this.IH.getRowDimension() + "," + this.IH.getColumnDimension() + "," + this.HO.getColumnDimension() + "\n");
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
		sb.append(v.getDimension());
		sb.append('\n'); 

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
	
	public static void run(){
		String input_filepath = "auto4.in"; 
		String output_filepath = "auto4.in"; 
		int h_size = 2; 
		double lambda = 0.01; 
		int max_epoch = 1000000; 
		
		RealMatrix inputs = BAL.loadFromFile(input_filepath);
		RealMatrix outputs = BAL.loadFromFile(output_filepath);
		
		BAL network = new BAL(inputs.getColumnDimension(), h_size, outputs.getColumnDimension()); 
		
		ArrayList<Integer> order = new ArrayList<Integer>(inputs.getRowDimension());
		for(int i=0; i<inputs.getRowDimension() ; i++){
			order.add(i); 
		}
		
		for(int e=0; e<max_epoch ; e++){
			java.util.Collections.shuffle(order);
			for(int order_i = 0; order_i < order.size() ; order_i++){
				RealVector in = inputs.getRowVector(order_i);
				RealVector out = outputs.getRowVector(order_i);
				network.learn(in, out, lambda); 
			}
			
			if(e % 100 == 0){
				//System.out.println(network.evaluate(inputs, outputs));
				network.measure(e, inputs, outputs);
			}
		}
		//System.out.println(network.printNetwork());
		
		network.saveMeasure("data/measure_" + ((int)network.evaluate(inputs, outputs)) + "_" + (System.currentTimeMillis() / 1000L) + ".dat");
	}
	
	public static void main(String[] args) {
		for(int i=0 ; i<1000 ; i++){
			BAL.run(); 
		}
	}
}
