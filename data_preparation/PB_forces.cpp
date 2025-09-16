#include <iostream>
using namespace std;
#include <cstring>
#include <math.h>
#include <stdlib.h> 
#include<fstream>
#include<istream>
#include<vector>
#include<sstream>
#include <typeinfo>
#include <assert.h> 
#include <algorithm>
#include <sstream>
#include <iterator>
#include <iomanip>
#include <numeric>



int main() {
    struct interpolar_input {int in_nx; int in_ny;int in_nz; double in_xmin; double in_xmax; double in_ymin; double in_ymax; double in_zmin; double in_zmax; double in_hx; double in_hy; double in_hz;};
  std::vector <interpolar_input> I;
  std::vector < std::vector < std::vector<double> > > test_x;
  std::vector < std::vector < std::vector<double> > > test_y;
  std::vector < std::vector < std::vector<double> > > test_z;
  

  vector <double> data_x;
  vector <double> data_y;
  vector <double> data_z;
  vector <double> sigma_LJ_atom;
    ifstream file_x("efield_x.dx");
    ifstream file_y("efield_y.dx");
    ifstream file_z("efield_z.dx");
    string line_x; 
    string line_y;
    string line_z;
    string snx, sxmin, shx;
    string sny, symin,shy;
    string snz, szmin,shz;
    string sefield_x;
    double efield_x;
    string sefield_y;
    double efield_y;
    string sefield_z;
    double efield_z;
    string sefield_zkonec;
    double efield_zkonec;
    double E_unit_kcal =0.596161774 ;//kb*T/ec T =300K
   
    while (getline(file_x, line_x)){
        vector <string> v;
        istringstream ss(line_x);
        string word;
        while (ss >> word) {
        v.push_back(word);
        }
        if (v[0] == "#"){continue;}
        
         if (v[0] == "object" && v[3] == "gridpositions"){
             snx = v[5];
             sny = v [6];
             snz = v[7];

        
        }
        if (v[0] == "origin"){
            sxmin = v[1];
            symin = v [2];
            szmin = v[3];
          
            
        }
        if (v[0] == "delta"){
            if (v[1] != "0.000000e+00"){
            shx = v[1];}
            if (v[2] != "0.000000e+00"){
            shy = v [2];}
            if (v[3] != "0.000000e+00"){
                shz = v[3];
                }
        }
       if (v[0] != "object" &&  v[0] != "origin" && v[0] != "delta"&& v[0] != "attribute"&& v[0] != "component"){
           for (int index = 0; index < v.size(); ++index) {
               
               sefield_x = v[index];
               stringstream gsefield(sefield_x);
               efield_x = 0.0;
               gsefield >> efield_x;
              
               data_x.push_back(efield_x);}} 
    }
    file_x.close();
    cout <<"file_closed"<<"size of data_x"<<data_x.size()<<endl;
    
    while (getline(file_y, line_y)){
        
        vector <string> v_y;
        v_y.clear();
        istringstream ss(line_y);
        string word_y;
        while (ss >> word_y) {
        v_y.push_back(word_y);
        }
        if (v_y[0] == "#"){continue;}
        if (v_y[0] != "object" &&  v_y[0] != "origin" && v_y[0] != "delta"&& v_y[0] != "attribute"&& v_y[0] != "component"){
        for (int index = 0; index < v_y.size(); ++index) {
              
               sefield_y = v_y[index];
               stringstream gsefield(sefield_y);
               efield_y = 0.0;
               gsefield >> efield_y;
               data_y.push_back(efield_y);}} 
    }
    file_y.close();
    // cout <<"file_closed"<<"size of data_y "<<data_y.size()<<endl;
    cout <<"file_closed"<<"size of data_y"<<data_y.size()<<endl;
   
    while (getline(file_z, line_z)){
        //cout <<elecfield_z<<"  in loop for open elecfield_z "<<endl;
        vector <string> v_z;
        v_z.clear();
        istringstream ss(line_z);
        string word_z;
        while (ss >> word_z) {
        v_z.push_back(word_z);
        }
        if (v_z[0] == "#"){continue;}
         if (v_z[0] != "object" &&  v_z[0] != "origin" && v_z[0] != "delta"&&  v_z[0] != "attribute" && v_z[0] != "component"){
           for (int index = 0; index < v_z.size(); ++index) {
              
               sefield_z = v_z[index];
               stringstream gsefield(sefield_z);
               efield_z = 0.0;
               gsefield >> efield_z;
               data_z.push_back(efield_z);}} 
    }

    file_z.close();
    cout <<"file_closed"<<"size of data_z"<<data_z.size()<<endl;
   
   
    stringstream gsnx(snx);
    stringstream gsny(sny);
    stringstream gsnz(snz);
    stringstream gsxmin(sxmin);
    stringstream gsymin(symin);
    stringstream gszmin(szmin);
    stringstream gshx(shx);
    stringstream gshy(shy);
    stringstream gshz(shz);
    int nx = 0;
    int ny=0;
    int nz=0;
    double xmin=0.0;
    double ymin=0.0;
    double zmin=0.0;
    double hx = 0.0;
    double hy = 0.0;
    double hz = 0.0;
    
  
    gsnx >> nx;
    gsny >> ny;
    gsnz >> nz;
    gsxmin >> xmin;
    gsymin >> ymin;
    gszmin >> zmin;
    gshx >> hx;
    gshy >> hy;
    gshz >> hz;
    

    double xmax=xmin+(double(nx-1))*hx;
    double ymax=ymin+(double(ny-1))*hy;
    double zmax=zmin+(double(nz-1))*hz;//+0.0001;
    cout <<"xmin "<<xmin<<"xmax "<<xmax<<"ymin "<<ymin<<"ymax "<<ymax<<"zmin "<<zmin<<"zmax "<<zmax<<endl;
    cout<<"hx "<<hx<<"hy "<<hy<<"hz "<<hz<<"nx "<<nx<<"ny "<<ny<<"nz "<<nz<<endl;
  
    I.push_back({nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, hx, hy, hz});
    cout <<"xmin "<<(I[0].in_xmin )<<"xmax "<<(I[0].in_xmax)<<"ymin "<<I[0].in_ymin<<"ymax "<<I[0].in_ymax<<"zmin "<<I[0].in_zmin<<"zmax "<<I[0].in_zmax<<endl;
    cout<<"hx "<<I[0].in_hx<<"hy "<<I[0].in_hy<<"hz "<<I[0].in_hz<<"nx "<<I[0].in_nx<<"ny "<<I[0].in_ny<<"nz "<<I[0].in_nz<<endl;
   
    //cout <<"here"<<endl;
    int countLines= 0;
    cout <<"here"<<endl;
    for (int i = 0; i < nx; ++i) {
        test_x.push_back(vector<vector<double>>());
        test_y.push_back(vector<vector<double>>());
        test_z.push_back(vector<vector<double>>());
       
        for (int j = 0; j < ny; ++j) {
            test_x[i].push_back(vector<double>());
            test_y[i].push_back(vector<double>());
            test_z[i].push_back(vector<double>());
           
            for (int k = 0; k <nz; ++k) {
                countLines++;
                test_x[i][j].push_back(data_x[countLines-1]);
                test_y[i][j].push_back(data_y[countLines-1]);
                test_z[i][j].push_back(data_z[countLines-1]);
               
            }
        }
    }

   

ofstream outfile;

outfile.open("tray_PB_forces.xyz");


ifstream input_file("tray_allatom.lammpstrj"); 
string line_input;


vector <double> v_Na[3];
vector <double> v[3];

int count_structures_loop = 0;  

while (getline(input_file, line_input)){
        vector <string> v_input;
    
        istringstream ss(line_input);
        string word;
        
        while (ss >> word) {
        v_input.push_back(word);
        }
        
        if (v_input[0] == "ITEM:" && v_input[1] == "TIMESTEP"){
            count_structures_loop += 1;
            cout<<"in loop "<<count_structures_loop<<endl;
        }
        
       
        if (v_input[0] != "ITEM:"){ 
            
        if ((v_input.size()) > 5 ){
       
        string sxpos, sypos, szpos;
        sxpos = v_input[3];
        sypos = v_input[4];
        szpos = v_input[5];
        
       
            
        stringstream gsxpos(sxpos);
        stringstream gsypos(sypos);
        stringstream gszpos(szpos);
       
        double x_pos = 0.0;
        double y_pos = 0.0;
        double z_pos = 0.0;
       
        gsxpos >> x_pos;
        gsypos >> y_pos;
        gszpos >> z_pos;
      
       
     
size_t ihi, jhi, khi, ilo, jlo, klo; 

double ifloat, jfloat, kfloat;
double u_x = 0.0;
double u_y = 0.0;
double u_z = 0.0;

double x_efield = 0.0;
double y_efield = 0.0;
double z_efield = 0.0;

ifloat = ((x_pos-(I[0].in_xmin))/I[0].in_hx); 
jfloat = ((y_pos-(I[0].in_ymin))/I[0].in_hy);
kfloat = ((z_pos-(I[0].in_zmin))/I[0].in_hz);








ihi = (int)ceil(ifloat);
ilo = (int)floor(ifloat);
jhi = (int)ceil(jfloat);
jlo = (int)floor(jfloat);
khi = (int)ceil(kfloat);
klo = (int)floor(kfloat);



int Vgrid_digits = 9; //tudi v APBS toliko 6
double Vcompare = pow(10,-1*(Vgrid_digits - 2));




double dx = (ifloat - double(ilo));
double dy = (jfloat - double(jlo));
double dz = (kfloat - double(klo));




 
int a1,a2,a3,a4,a5,a6,a7,a8;
int b1,b2,b3,b4,b5,b6,b7,b8;
int c1,c2,c3,c4,c5,c6,c7,c8;

a1 = a2 = a3 = a4 = ihi;
a5 = a6=a7=a8 = ilo;
b1 = b3 = b5 = b7 = jhi;
b2 = b4=b6=b8 = jlo;
c1=c2=c5=c6= khi;
c3=c4=c7=c8 = klo;

u_x =(dx      *dy      *dz      *test_x[a1][b1][c1]
    + dx      *(1.0-dy)*dz      *test_x[a2][b2][c2]
    + dx      *dy      *(1.0-dz)*test_x[a3][b3][c3]
    + dx      *(1.0-dy)*(1.0-dz)*test_x[a4][b4][c4]
    + (1.0-dx)*dy      *dz      *test_x[a5][b5][c5]
    + (1.0-dx)*(1.0-dy)*dz      *test_x[a6][b6][c6]
    + (1.0-dx)*dy      *(1.0-dz)*test_x[a7][b7][c7]
    + (1.0-dx)*(1.0-dy)*(1.0-dz)*test_x[a8][b8][c8]);

u_y =(dx      *dy      *dz      *test_y[a1][b1][c1]
    + dx      *(1.0-dy)*dz      *test_y[a2][b2][c2]
    + dx      *dy      *(1.0-dz)*test_y[a3][b3][c3]
    + dx      *(1.0-dy)*(1.0-dz)*test_y[a4][b4][c4]
    + (1.0-dx)*dy      *dz      *test_y[a5][b5][c5]
    + (1.0-dx)*(1.0-dy)*dz      *test_y[a6][b6][c6]
    + (1.0-dx)*dy      *(1.0-dz)*test_y[a7][b7][c7]
    + (1.0-dx)*(1.0-dy)*(1.0-dz)*test_y[a8][b8][c8]);
        
u_z =(dx      *dy      *dz      *test_z[a1][b1][c1]
    + dx      *(1.0-dy)*dz      *test_z[a2][b2][c2]
    + dx      *dy      *(1.0-dz)*test_z[a3][b3][c3]
    + dx      *(1.0-dy)*(1.0-dz)*test_z[a4][b4][c4]
    + (1.0-dx)*dy      *dz      *test_z[a5][b5][c5]
    + (1.0-dx)*(1.0-dy)*dz      *test_z[a6][b6][c6]
    + (1.0-dx)*dy      *(1.0-dz)*test_z[a7][b7][c7]
    + (1.0-dx)*(1.0-dy)*(1.0-dz)*test_z[a8][b8][c8]);
            
       

x_efield = -u_x * E_unit_kcal;
y_efield = -u_y * E_unit_kcal;
z_efield = -u_z * E_unit_kcal;
        




if (v_input[2] == "24"){
 

    v[0].push_back(x_efield*(-1));
    v[1].push_back(y_efield*(-1));
    v[2].push_back(z_efield*(-1));
      
}




if (v_input[2] == "25"){
  

    v_Na[0].push_back(x_efield*(1));
    v_Na[1].push_back(y_efield*(1));
    v_Na[2].push_back(z_efield*(1));
    
}

 }}}






ifstream input_file1("tray_allatom.lammpstrj");
string line_input1;


int count = 0;
int count_Na = 0;
int count_Cm = 0;

while (getline(input_file1, line_input1)){
        vector <string> v_input;
        istringstream ss(line_input1);
        string word;
        
        while (ss >> word) {
        v_input.push_back(word);
        }
         if ((v_input[0] == "ITEM:" && v_input[1] == "TIMESTEP")){
            count_Cm += 1;
           
        }
         
          
          if (v_input[0] != "ITEM:" && ((v_input.size() > 5 ))){
             
              
           
            if (v_input[2] == "24"){
              
            count += 1;
           
            vector <double> list;
            
            for (int sym = 0; sym < 3; sym++){
            std::stringstream ss;
            ss << std::fixed << std::setprecision(9) << v[sym][count-1];
            std::string s_scaled_vale_1 = ss.str();
            
            v_input.push_back(s_scaled_vale_1);
            cout<<s_scaled_vale_1<<endl;
                
            }}
             
            
             
             
             
             
             
            if (v_input[2] == "25"){
               
            count_Na += 1;
            
            vector <double> list;
            
            for (int sym = 0; sym < 3; sym++){
            std::stringstream ss;
            ss << std::fixed << std::setprecision(9) << v_Na[sym][count_Na-1];
            std::string s_scaled_vale_1_Na = ss.str();    
           
            v_input.push_back(s_scaled_vale_1_Na);

            }}
          }

for(int i=0;i<v_input.size();++i){
		outfile<<v_input[i]<<" ";}
		outfile<<endl;
}


input_file1.close();

outfile.close();

}
