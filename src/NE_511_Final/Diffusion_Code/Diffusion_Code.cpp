#define _USE_MATH_DEFINES

#include <cmath>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <boost/math/special_functions/bessel.hpp> 
using namespace Eigen;

int N1 = 10;
int N2 =  0;
int Nt = N1 + N2;

// ------ Define Geometry ----------
double R1 = 0; // If R1=0 one region
double R2 = 4;
bool region = false; // If region=true two regions, if region=false one region

// ------ Define Nuclear Data ------
double D = 2;
double Sig_a = 1;
double nuSig_f = 0.75;


auto get_rc = [] ( auto& R1, auto& R2, auto& N ) {
		VectorXd rc;
		rc.setLinSpaced( N, R2/(2*N), R2 );
		return rc;
	      };

auto get_rb = [] ( auto& R1, auto& R2, auto& N ) {
		VectorXd rb;
		rb.setLinSpaced( N, R1, R2-R2/(2*N) );
		return rb;
	      };

auto get_A = [&] ( auto& N, auto& R1, auto& R2, auto& D, auto& Sig_a ) {
	       VectorXd rb = get_rb( R1, R2, N );
	       VectorXd rc = get_rc( R1, R2, N );
	       MatrixXd A(N-1,N-1);
	       
	       A(0,0) = D*rb(1)/( rc(0) * ( rc(1) - rc(0) ) * ( rb(1) - rb(0) ) ) + Sig_a;
	       A(0,1) = (-1)*D*rb(1)/( rc(0) * ( rc(1) - rc(0) ) * ( rb(1) - rb(0) ) );
	       A(N-2,N-2) = D*rb(N-1)/( rc(N-2) * ( rc(N-1) - rc(N-2) ) * ( rb(N-1) - rb(N-2) ) ) + Sig_a
		 + D*rb(N-2)/( rc(N-2) * ( rc(N-2) - rc(N-3) ) * ( rb(N-2) - rb(N-3) ) );
	       A(N-2,N-3) = (-1)*D*rb(N-2)/( rc(N-2) * ( rc(N-2) - rc(N-3) ) * ( rb(N-2) - rb(N-3) ) );
	       
	       for( int i=1; i<N-2; ++i ){
		 A(i,i) = D*rb(i+1)/( rc(i) * ( rc(i+1) - rc(i) ) * ( rb(i+1) - rb(i) ) ) + Sig_a
		   + D*rb(i)/( rc(i) * ( rc(i) - rc(i-1) ) * ( rb(i) - rb(i-1) ) ); 
		 A(i,i+1) = (-1)*D*rb(i+1)/( rc(i) * ( rc(i+1) - rc(i) ) * ( rb(i+1) - rb(i) ) );
		 A(i,i-1) = (-1)*D*rb(i)/( rc(i) * ( rc(i) - rc(i-1) ) * ( rb(i) - rb(i-1) ) );
	       }
	       
	       return A;
	     };

auto setup = [&] ( auto& N1, auto& N2, auto& R1, auto& R2, auto& reg, auto& D, auto& Sig_a ) {
	       MatrixXd A;
	       if (!reg) {
		 A = get_A( N1, R1, R2, D, Sig_a );
	       }
	       return A;
	     };

auto gen_b = [] ( VectorXd phi, VectorXd& b, double& nuSig_f, double& k, int& N ) {
	       for( int i=0; i<N-1; ++i ){
		 b(i) = (1/k)*nuSig_f*phi(i);
	       }
	     };

auto calc_k = [] ( VectorXd phi_n, VectorXd phi_o, VectorXd Rc, auto& nuSig_f, auto k_o, auto& N ) {
		double numerator = 0.0;
		double denominator = 0.0;
		for( int i = 0; i<N-1; ++i ){
		  numerator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )*nuSig_f*phi_n(i);
		  denominator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )*nuSig_f*phi_o(i);
		}
		double k_n;
		return k_n = k_o*(numerator/denominator);
	      };

auto init_phi = [] ( auto& N ) {
		  VectorXd phi(N-1);
		  for( int i=0; i<N-1; ++i ){
		    phi(i) = 1.0;
		  }
		  return phi;
		};

auto printer = [] ( auto& Rc, auto& phi, auto& N ) {
		 std::ofstream myfile ("output.txt");
		 if (myfile.is_open()) {
		   for( int i = 0; i < N-1; ++i ){
		     myfile << Rc(i) << " " <<  phi(i) << "\n";
		   }
		 }
		 myfile.close();
	       };

		 
int main()
{
  // int v = 0;
  // double z = 2;
  // auto r = cyl_bessel_i(v, z);

  // ---- Setup Problem ----------------------------------------------------
  MatrixXd A  = setup( N1, N2, R1, R2, region, D, Sig_a );
  VectorXd Rc = get_rc( R1, R2, N1 );

  ColPivHouseholderQR<MatrixXd> dec(A);
  VectorXd phi = init_phi( Nt );
  VectorXd phi_n = init_phi( Nt );
  double k = 1.0;

  VectorXd b(Nt - 1);
  gen_b( phi, b, nuSig_f, k, Nt );
  
  for( int i = 0; i < 10; ++i ){
    gen_b( phi, b, nuSig_f, k, Nt );
    phi_n = dec.solve(b);
    k = calc_k( phi_n, phi, Rc, nuSig_f, k, Nt );
    phi = phi_n;
  }

  // printer( Rc, phi );
  
  std::cout << k << std::endl;
}

