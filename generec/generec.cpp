//tail -n +2 train.csv | awk 'BEGIN{FS=","}{for(i=0;i<10;i++) {if($1 == i) printf "1 "; else printf "0 ";} print ""}' > digit.train

//Hidden: 300 Epoch: 5 Alpha: 0.1 Success: 92.2167

//Success rate 38919/42000 92.6643%
//Hidden: 300 Epoch: 10 Alpha: 0.05

//Success rate 38380/42000 91.381%
//Hidden: 30 Epoch: 10 Alpha: 0.10

//Success rate 16686/42000 39.7286%
//Hidden: 10 Epoch: 10 Alpha: 0.20

//Success rate 18236/42000 43.419%
//Hidden: 10 Epoch: 10 Alpha: 0.10

//Success rate 17709/42000 42.1643%
//Hidden: 10 Epoch: 10 Alpha: 0.05

//Success rate 17812/42000 42.4095%
//Hidden: 10 Epoch: 30 Alpha: 0.075

//Success rate 39321/42000 93.6214%
//Hidden: 300 Epoch: 50 Alpha: 0.05

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
        
        REP(a, rows){
          REP(b, cols){
            fimg << min(255, max(0, (int)(64 + 4 * (this->at(a*cols + b))))) << " ";
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
    
    void print(ofstream& fout){
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




int main(int argc, char *argv[])
{
  cout << "start" << endl;
  VectorSet I(argv[1]);
  VectorSet O(argv[2]); 
  Uint Hsize = atoi(argv[3]);
  Uint epochs = atoi(argv[4]);
  Double epsilon = atof(argv[5]); 
  
  Matrix WIH(I.at(0)->size(),Hsize);
  Matrix WHO(Hsize,O.at(0)->size());   
  cout << "init ok" << endl;
  
  REP(e, epochs){
    cout << "epoch " << e+1 << endl;
  
    REP(in, I.size()){ 
      if((in)%1000==0) cout << "  input " << in+1 << endl;
    
      Vector* hn_minus = WIH.multiplyLeft(I.at(in)); //cout << "hn_minus ok " << hn_minus << endl;
      Vector* hminus = hn_minus->applyToNew(Sigmoid);  //cout << "minus ok" << endl; 
      Vector* on_minus = WHO.multiplyLeft(hminus);  //cout << "on_minus ok" << endl; 
      Vector* ominus = on_minus->applyToNew(Sigmoid);  //cout << "ominus ok" << endl;

      Vector* hn_plus_first = WIH.multiplyLeft(I.at(in));  //cout << "hn_plus_first ok" << endl; 
      Vector* hn_plus_second = WHO.multiplyRight(O.at(in));  //cout << "hn_plus_second ok" << endl; 
      Vector* hn_plus = hn_plus_first->add(hn_plus_second);  //cout << "hn_plus ok" << endl; 
      Vector* hplus = hn_plus->applyToNew(Sigmoid);  //cout << "hplus ok" << endl; 

/*/
      cout << "IN " << in << endl;
      I.at(in)->print(); 
      
      cout << "WIH " << endl;
      WIH.print();
      cout << "hn_minus " << endl;       
      hn_minus->print(); 
      cout << "hminus " << endl;       
      hminus->print();
      cout << "WHO " << endl;       
      WHO.print();
      cout << "on_minus " << endl;        
      on_minus->print();
      
      cout << "ominus " << endl;       
      ominus->print();
      cout << "hplus " << endl;       
      hplus->print();
      
      cout << endl << "Before WIH " << endl;
      WIH.print(); 
      REP(i, WIH.rows()) REP(j, WIH.cols()){
        WIH.set(i, j, WIH.at(i,j) + epsilon*I.at(in)->at(i) * (hplus->at(j) - hminus->at(j))); 
      }
      cout << "After WIH " << endl;
      WIH.print(); 
      cout << endl << "Before WHO " << endl;
      WHO.print(); 
      REP(i, WHO.rows()) REP(j, WHO.cols()){
        WHO.set(i, j, WHO.at(i,j) + epsilon*hminus->at(i) * (O.at(in)->at(j) - ominus->at(j))); 
      }
      cout << "After WHO " << endl;
      WHO.print(); 
      /*/
      REP(i, WIH.rows()) REP(j, WIH.cols()){
        WIH.set(i, j, WIH.at(i,j) + epsilon*I.at(in)->at(i) * (hplus->at(j) - hminus->at(j))); 
      }
      REP(i, WHO.rows()) REP(j, WHO.cols()){
        WHO.set(i, j, WHO.at(i,j) + epsilon*hminus->at(i) * (O.at(in)->at(j) - ominus->at(j))); 
      }/**/
      //cout << "==SUMMARY " << in << endl;
      //I.at(in)->print(); 
      //ominus->print();
      //O.at(in)->print(); 

      //cout << "deleting" << endl;
      delete hn_minus; //cout << "hn_minus ok" << endl; 
      delete hminus; //cout << "hminus ok" << endl; 
      delete on_minus; //cout << "on_minus ok" << endl;  
      delete ominus; //cout << "ominus ok" << endl; 
      
      delete hn_plus_first; //cout << "hn_plus_first ok" << endl; 
      delete hn_plus_second; //cout << "hn_plus_second ok" << endl; 
      delete hn_plus; //cout << "hn_plus ok" << endl; 
      delete hplus; //cout << "hplus ok" << endl; 
      //cout << "delete ok" << endl;
    } 
  }
  cout << "train ok" << endl;
  cout << "argc " << argc << endl;
  cout << "testing" << endl;
 
  //TEST
  if(argc > 6){
    VectorSet tI(argv[6]);
    VectorSet tO(argv[7]); 
    ofstream fout(argv[8]); 
    
    WIH.print(fout);
    WHO.print(fout); 
    
    Uint suc = 0; 
    
    REP(t, tI.size()){
        if((t)%1000==0) cout << "  test " << t+1 << endl;
        Vector* hn_minus = WIH.multiplyLeft(tI.at(t));  //cout << "hn_minus ok " << hn_minus << endl;
        Vector* hminus = hn_minus->applyToNew(Sigmoid);  //cout << "minus ok" << endl; 
        Vector* on_minus = WHO.multiplyLeft(hminus);  //cout << "on_minus ok" << endl; 
        Vector* ominus = on_minus->applyToNew(Sigmoid);  //cout << "ominus ok" << endl;
        
        int max_id = 0;
        REP(i, ominus->size()) if(ominus->at(i) > ominus->at(max_id)) {max_id=i;}
        
        suc += (tO.at(t)->at(0) == max_id); 
        fout << max_id << endl;
    }
    
    cout << "Success rate " << suc << "/" << tI.size() << " " << 100.0*((Double)suc/(Double)tI.size()) << "%" << endl; 
    cout << "Hidden: " << argv[3] << " Epoch: " << argv[4] << " Alpha: " << argv[5] << endl;
    
    
    REP(i, 10){
      cout << "creating backward activation image for " << i << endl;
      
      Vector hidden_i = WHO.getColumnAsVector(i); 
      Vector* img_i = WIH.multiplyRight(&hidden_i);
      
      string digit = "0";
      digit[0] += i; 
      img_i->saveAsPgm(string(argv[9]) + "/dig_" + digit + ".pgm", 28, 28);
      
      cout << "done" << endl;
    }
    
    REP(i, WIH.cols()){
      string digit = "0";
      digit[0] += i; 
      Vector feature_i = WIH.getColumnAsVector(i);
      feature_i.saveAsPgm(string(argv[9]) + "feat_" + digit + ".pgm", 28, 28);
    }
    
    
    fout.close(); 
  }
 
  cout << "clean up ok" << endl;
  
  return 0;
}
