//tail -n +2 train.csv | awk 'BEGIN{FS=","}{for(i=0;i<10;i++) {if($1 == i) printf "1 "; else printf "0 ";} print ""}' > digit.train

//Hidden: 300 Epoch: 5 Alpha: 0.1 Success: 92.2167

//Success rate 38380/42000 91.381%
//Hidden: 30 Epoch: 10 Alpha: 0.10

//Success rate 16686/42000 39.7286%
//Hidden: 10 Epoch: 10 Alpha: 0.20

//Success rate 38130/42000 90.7857%
//Hidden: 30 Epoch: 10 Alpha: 0.075

//Success rate 17569/42000 41.831%
//Hidden: [10] Epoch: 5 Alpha: 0.075

//Success rate 8036/42000 19.1333%
//Hidden: 20 10
//Epoch: 5 Alpha: 0.075

//Success rate 12224/42000 29.1048%
//Hidden: 200 40 
// Epoch: 10 Alpha: 0.03

//Success rate 38130/42000 90.7857%
//Hidden: 30 
//Epoch: 10 Alpha: 0.075

//Success rate 38380/42000 91.381%
//Hidden: 30 
//Epoch: 10 Alpha: 0.1

//Success rate 18153/42000 43.2214%
//Hidden: 300 50 
// Epoch: 50 Alpha: 0.03

#include <iostream>
#include <vector> 
#include <ctime> 
#include <string> 
#include <fstream>
#include <cstdlib>
#include <cmath>

using namespace std;

typedef double Double; 
typedef unsigned int Uint; 

#define REP(i, to) for(int i=0; i<to; i++)
Double randDouble(Double from, Double to){
  Double length = to - from; 
  Double r = (Double) (rand() % 100000);
  return from + ((Double)r / 100000.0f) * length;  
}

typedef Double (*Function)(Double); 

Double Sigmoid(Double param){
  return (1 / (1 + exp(-param))); 
}

class Vector{
  protected: 
    vector<Double> v; 

  public:
    Vector(Uint size){
      v.resize(size); 
    }/*
    ~Vector(){
      cout << "Vector delete " << size() << " " << this << endl;
      delete v; 
    }*/
    
    Uint size(){
      return v.size(); 
    }
    Double at(Uint i){
      return v[i]; 
    }
    void set(Uint i, Double val){
      v[i] = val; 
    }
    void apply(Function f){
      if(f==NULL) {
        cerr << "Apply with NULL parameter" << endl;
        return; 
      }
      REP(i, size()) {v[i] = f(v[i]); } 
    }
    Vector* copy(){
      Vector* result = new Vector(size()); 
      REP(i, size()) result->set(i, v[i]); 
      return result; 
    }
    Vector* applyToNew(Function f){
      Vector* result = copy(); //cout << "copy";
      result->apply(f);  //cout << "apply";
      return result; 
    }
    void print(){
      //cout << "Vector: " << size() << endl;
      REP(i, size()) cout << v[i] << " "; 
      cout << endl;
    }
    
    Vector* add(Vector* other){
      if(size() != other->size()) return NULL; 
      Vector* result = new Vector(size()); 
      REP(i, size()) result->set(i, v[i] + other->at(i)); 
      return result; 
    }
    
    
  void saveAsPgm(string filename, int rows, int cols){
        ofstream fimg(filename.c_str());
        fimg << "P2" << endl;
        fimg << rows << " " << cols << endl;
        fimg << "256" << endl;
        
        //Double in = 123456789;
        Double ax = -123456789;
        REP(i, size()){
          //in = min(in, at(i)); 
          ax = max(ax, abs(at(i)));           
        }
        
        REP(a, rows){
          REP(b, cols){
            fimg << (int)((abs(at(a*cols + b)) / ax) * 256.0) << " ";
          }
          fimg << endl;
        }
        
        fimg.close(); 
  }
}; 

class Matrix{
  protected: 
    Double** m; 
    Uint dimx;
    Uint dimy; 

  public: 
  //x - rows
  //y - columns
    Matrix(Uint dimension_x, Uint dimension_y){
      cout << "Matrix create " << dimension_x << " " << dimension_y << endl;
      dimx = dimension_x;
      dimy = dimension_y; 
      m = new Double* [dimx];
      REP(i, dimx)
        m[i] = new Double[dimy];
        
      REP(i, rows()) REP(j, cols()) m[i][j] = randDouble(-1.0, 1.0);
    }
    ~Matrix(){
      //cout << "Matrix delete " << dimx << " " << dimy << endl;
      //REP(i, dimx) 
      //  delete [] m[i];
      //delete [] m;
    }
    
    Uint rows() {return dimx;} 
    Uint cols() {return dimy;} 
    Double at(Uint i, Uint j) {return m[i][j];}
    Double set(Uint i, Uint j, Double val) {m[i][j] = val;} 
    
    //as row vector * matrix
    Vector* multiplyLeft(Vector* v){
      if(v->size() != rows()) {
        cout << "MultiplyLeft not working ["<<rows()<<","<<cols()<<"] and " << v->size() << endl;
        cout << "  matrix: " << this << " vector: " << v << endl;
        return NULL; 
      }
      Vector* result = new Vector(cols()); 
      REP(i, cols()){
        Double sum = 0;
        REP(j, rows()) sum += m[j][i] * v->at(j);
        result->set(i, sum); 
      }
      return result; 
    }//as matrix * column vector
    Vector* multiplyRight(Vector* v){
      if(v->size() != cols()) {
        cout << "MultiplyRight not working ["<<rows()<<","<<cols()<<"] and " << v->size() << endl;
        return NULL; 
      }
      Vector* result = new Vector(rows()); 
      REP(i, rows()){
        Double sum = 0;
        REP(j, cols()) sum += m[i][j] * v->at(j);
        result->set(i, sum); 
      }
      return result; 
    }
    
    Vector getColumnAsVector(int i){
      Vector result(rows());
      REP(j, rows()){
        result.set(j, at(i, j)); 
      }
      return result; 
    }
    
    void print(ostream& fout){
      fout << "Matrix: " << rows() << " " << cols() << endl;
      REP(i, rows()){
        REP(j, cols()) fout << m[i][j] << " ";
        fout << endl;
      }
    }
}; 

class VectorSet{
  protected: 
    vector<Vector*> s; 
    
  public: 
    VectorSet(string filename){
      cout << "VectorSet create " << filename << endl;
      Uint x, y;
      ifstream fin(filename.c_str()); 
      fin >> x >> y;
      cout << " of size " << x << " " << y << endl;
      
      REP(i, x){
        Vector* v = new Vector(y);
        REP(j, y){
          Double val;
          fin >> val;
          v->set(j, val); 
        }
        s.push_back(v); 
      }
      fin.close(); 
    }
    ~VectorSet(){
      //cout << "VectorSet delete " << endl;
      REP(i, s.size()) delete s[i];
    }
    
    Uint size() {return s.size();}     
    Vector* at(Uint i) {return s[i];}
};

void coutVector(vector<Uint>& V){
  cout << "["; 
  REP(i, V.size()){
    cout << V[i] + ", ";
  }
  cout << "]";  
}

class DeepGeneRec{
  protected:
    VectorSet* I;
    VectorSet* O; 
    vector<Uint> hls; 
    vector<Matrix*> W; 
      
  public:
  //@precondition input->size() == output->size()
    DeepGeneRec(VectorSet* input, VectorSet* output, vector<Uint> hidden_layer_sizes){
      cout << "DeepGeneRec create" << endl;
    
      I = input;
      O = output;
    
      hls.resize(hidden_layer_sizes.size() + 2, 0);
      
      hls[0] = input->at(0)->size();
      REP(i, hidden_layer_sizes.size()) hls[i+1] = hidden_layer_sizes[i];
      hls[hls.size()-1] = output->at(0)->size(); 
      
      REP(i, hls.size()-1){
        W.push_back(new Matrix(hls[i], hls[i+1])); 
      }
      
      REP(i, hls.size()) cout << "  " << hls[i] << endl;
    }
    
    ~DeepGeneRec(){
      delete I;
      delete O;
      REP(i, W.size()) delete W[i];
    }
    
    //result[0] == in (therefore do not delete)
    //size == W.size() + 1 
    vector<Vector*> forwardPass(Vector* in){
      vector<Vector*> result; 
      result.push_back(in); 
      
      REP(i, W.size()){
        result.push_back(W[i]->multiplyLeft(result[i]));
      }
      
      return result; 
    }
    
    //result[W.size()] == out (therefore do not delete)
    //size == W.size() + 1  
    vector<Vector*> backwardPass(Vector* out){
      vector<Vector*> result(W.size() + 1, NULL); 
      result[W.size()] = out; 
      
      REP(i, W.size()){
        result[W.size() - i - 1] = W[W.size() - i - 1]->multiplyRight(result[W.size() - i]); 
      }
      
      return result; 
    }
    
    //result[0] == in (therefore do not delete) 
    vector<Vector*> minusPhase(Vector* in){
      vector<Vector*> result; 
      result.push_back(in); 
      
      REP(i, W.size()){
        Vector* hn_minus = W[i]->multiplyLeft(result[i]); 
        result.push_back(hn_minus->applyToNew(Sigmoid));
        
        delete hn_minus; 
      }
      
      return result;
    }
    
    //result[W.size()] == out (therefore do not delete) 
    vector<Vector*> plusPhase(Vector* in, Vector* out){
      vector<Vector*> result(W.size() + 1, NULL);
      vector<Vector*> forward = forwardPass(in); 
      vector<Vector*> backward = backwardPass(out); 
      
      result[W.size()] = out; 
      
      for(int i=1; i < W.size() ; i++){
          //cout << i << " fs:" << forward[i]->size() << " bs: " << backward[i]->size() << endl; 
          Vector* hn_plus_first = forward[i];  //cout << "hn_plus_first ok" << endl; 
          Vector* hn_plus_second = backward[i];  //cout << "hn_plus_second ok" << endl; 
          Vector* hn_plus = hn_plus_first->add(hn_plus_second);  //cout << "hn_plus ok" << endl; 
          result[i] = hn_plus->applyToNew(Sigmoid);  //cout << "hplus ok" << endl; 
          
          delete hn_plus; 
      }
      
      deleteForwardPass(forward);
      deleteBackwardPass(backward); 
      
      return result; 
    }
    
    void deleteForwardPass(vector<Vector*>& pass){
      for(int i=1; i<=W.size() ; i++) delete pass[i];
    }
    void deleteBackwardPass(vector<Vector*>& pass){
      for(int i=0; i<W.size() ; i++) delete pass[i];
    }
    void deleteMinusPhase(vector<Vector*>& pass){
      for(int i=1; i<=W.size() ; i++) delete pass[i];
    }
    void deletePlusPhase(vector<Vector*>& pass){
      for(int i=1; i<W.size() ; i++) delete pass[i];
    }
    
    int Train(Uint epochs, Double epsilon){
      cout << "Train started" << endl;
        
      REP(e, epochs){
        cout << "================epoch " << e+1 << "================" << endl;
      
        REP(in, I->size()){ 
          if((in+1)%10000==0) {
            cout << "  input " << in+1 << endl;
          }
          /*
          if(I->at(in)->size() != 784){
            cout << "pipky " << in << " ";
            I->at(in)->print(); 
            continue;
          }*/
        
          vector<Vector*> minus = minusPhase(I->at(in)); 
          //cout << "mp ok ["<<minus.size()<<"]" << endl;
          //REP(i, minus.size()) cout << minus[i] << endl;
          
          vector<Vector*> plus = plusPhase(I->at(in), O->at(in)); 
          //cout << "pp ok ["<<plus.size()<<"]" << endl;
          //REP(i, minus.size()) cout << plus[i] << endl;

        //LEARN (experimental)
        
          for(int w=1; w <= W.size() ; w++){
            //cout << w << " ["<< W[w-1]->rows() << "," << W[w-1]->cols() << "]" << endl;
            //cout << minus[w-1]->size() << " " << plus[w]->size() << " " << minus[w]->size() << endl;
             
            REP(i, W[w-1]->rows()) REP(j, W[w-1]->cols()){
              //cout << "set " << i << "," << j << " from: " << W[w-1]->at(i, j);
              
              //TODO zlepsit algoritmus
              W[w-1]->set(i, j, W[w-1]->at(i,j) + epsilon*minus[w-1]->at(i) * (plus[w]->at(j) - minus[w]->at(j))); 
              
              //cout << " to: " << W[w-1]->at(i, j) << endl;
              //cout << "eps: " << epsilon << " minus: " << minus[w-1]->at(i) << " plus: " << plus[w]->at(j) << " minus_next: " << minus[w]->at(j) << endl;
            }
          }
          //cout << "learn ok" << endl;
          
          /*
          REP(i, minus.size()) {
            cout << "minus " << i << ": ";
            minus[i]->print(); 
          }
          REP(i, plus.size()) if(plus[i] != NULL) {
            cout << "plus " << i << ": ";
            plus[i]->print();
          }
          REP(w, W.size()) W[w]->print(cout); */
          
        //CLEANUP
          deleteMinusPhase(minus);
          deletePlusPhase(plus);
        } 
      }
      
      return 0;
    }
    
   void Test(VectorSet* tI, VectorSet* tO, ofstream& fout){
      Uint suc = 0; 
      
      fout << tI->size() << " " << 1 << endl;
      
      REP(t, tI->size()){
          if((t+1)%10000==0) cout << "  test " << t+1 << endl;
          
          vector<Vector*> minus = minusPhase(tI->at(t)); 
          Vector* ominus = minus[minus.size()-1];
          
          //distributed to scalar
          int max_id = 0;
          REP(i, ominus->size()) if(ominus->at(i) > ominus->at(max_id)) {max_id=i;}
          
          suc += (tO->at(t)->at(0) == max_id); 
          fout << max_id << endl;
      }
      
      REP(w, W.size()){
        W[w]->print(fout);
      }
      
      cout << "Success rate " << suc << "/" << tI->size() << " " << 100.0*((Double)suc/(Double)tI->size()) << "%" << endl; 
   }
   
   void BackwardImages(string dirname){
      int osize = O->at(0)->size();
      REP(i, osize){
        cout << "creating backward activation image for " << i << endl;
        
        Vector* out = new Vector(osize);
        REP(j, out->size()) out->set(j, 0); 
        out->set(i, 1); 
        vector<Vector*> backward = backwardPass(out); 
        
        string digit = "0";
        digit[0] += i; 
        backward[0]->saveAsPgm(dirname + "/dig_" + digit + ".pgm", 28, 28);
        
        deleteBackwardPass(backward); 
        
        cout << "done" << endl;
      }
      
      //top layer feature
      REP(i, W[0]->cols()){
        string digit = "0";
        digit[0] += i; 
        Vector feature_i = W[0]->getColumnAsVector(i);
        feature_i.saveAsPgm(dirname + "feat_" + digit + ".pgm", 28, 28);
      }
   }
};

//argv
//1: input data file
//2: train data file
//3: epoch count
//4: epsilon
//5: number of hidden layers (min 1)
//6-(6+$5): hidden layer sizes

int main(int argc, char *argv[])
{
  cout << "start" << endl;
  
  int epochs = atoi(argv[3]);
  Double epsilon = atof(argv[4]); 
  int hlc = atoi(argv[5]); 
  vector<Uint> hls; 
  REP(i, hlc) hls.push_back(atoi(argv[6+i])); 
  
  DeepGeneRec DGR(new VectorSet(string(argv[1])), new VectorSet(string(argv[2])), hls);
  
  cout << "init ok" << endl;
  
  DGR.Train(epochs, epsilon);
  
  cout << "train ok" << endl;
 
  //TEST
  int test_argc = 5+hls.size(); 
  if(argc > test_argc + 1){
    cout << "testing" << endl;
  
    VectorSet tI(argv[test_argc + 1]);
    VectorSet tO(argv[test_argc + 2]); 
    ofstream fout(argv[test_argc + 3]); 
    
    DGR.Test(&tI, &tO, fout); 
    
    cout << "Hidden: ";
    REP(i, hls.size()) cout << hls[i] << " ";
    cout << endl;
    cout << " Epoch: " << epochs << " Alpha: " << epsilon << endl;
  }
   
  DGR.BackwardImages(argv[test_argc + 4]); 
  
  return 0;
}
