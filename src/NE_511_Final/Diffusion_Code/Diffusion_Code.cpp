#define _USE_MATH_DEFINES

#include <cmath>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;

int N = 100;

// ------ Define Geometry ----------
double Rmin = 0;
double Rmax = 4;

// ------ Define Nuclear Data ------
double D = 2;
double Sig_a = 1;
double nuSig_f = 0.75;

VectorXd get_Rc( double& Rmin, double& Rmax, int& N ) {
  double delta = Rmax/N;
  VectorXd rc;
  rc.setLinSpaced( N, delta/2, Rmax );
  return rc;
}

VectorXd get_Rb( double& Rmin, double& Rmax, int& N ) {
  double delta = Rmax/N;
  VectorXd rb;
  rb.setLinSpaced( N, 0, Rmax-delta/2 );
  return rb;
}

MatrixXd get_A( int& N, VectorXd rc, VectorXd rb, double& D, double& Sig_a ) {
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
}

void get_b( VectorXd phi, VectorXd& b, double& nuSig_f, double& k, int& N ) {
  for( int i=0; i<N-1; ++i ){
    b(i) = (1/k)*nuSig_f*phi(i);
  }
}

double get_k( VectorXd phi_n, VectorXd phi_o, VectorXd Rc, double& nuSig_f, double k_o ) {
  double numerator = 0.0;
  double denominator = 0.0;
  for( int i = 0; i<N-1; ++i ){
    numerator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )*nuSig_f*phi_n(i);
    denominator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )*nuSig_f*phi_o(i);
  }
  double k_n;
  return k_n = k_o*(numerator/denominator);
}


int main()
{
  // ---- Setup Problem ----------------------------------------------------
  VectorXd Rc = get_Rc( Rmin, Rmax, N );
  VectorXd Rb = get_Rb( Rmin, Rmax, N );

  VectorXd phi(N-1);
  for( int i=0; i<N-1; ++i ){
    phi(i) = 0.5;
  }
  double k = 0.8;
  VectorXd b(N-1);
  b.setZero();

  MatrixXd A = get_A( N, Rc, Rb, D, Sig_a );
  //ColPivHouseholderQR<MatrixXd> dec(A);
  FullPivLU<Ref<MatrixXd> > lu(A);

  // ---- Solve Problem ---------------------------------------------------
  for( int i = 0; i < 10; ++i ){
    get_b( phi, b, nuSig_f, k, N );

    VectorXd x = lu.solve(b);
    k =  get_k( x, phi, Rc, nuSig_f, k );
    //phi = dec.solve(b);
    phi = x;
  }

  std::ofstream myfile ("output.txt");
  if (myfile.is_open())
    {
      for( int i = 0; i < N-1; ++i ){
	myfile << Rc(i) << " " <<  phi(i) << "\n";
      }
    }
  myfile.close();
  std::cout << k << std::endl;
  // std::cout << "-------------" << std::endl;
  // for( int i=0; i<N; ++i ){
  //   std::cout << Rc(i) << ", " << phi(i) << std::endl;
  // }
}

