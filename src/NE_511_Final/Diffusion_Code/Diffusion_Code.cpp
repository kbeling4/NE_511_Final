#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;

int N = 10;

// ------ Define Geometry ----------
double Rmin = 0;
double Rmax = 2;
double delta = Rmax/N;

// ------ Define Nuclear Data ------
double D = 1;
double Sig_a = 1;
double Sig_f = 0.5;


int main()
{
  VectorXf rc;
  VectorXf rb;
  rc.setLinSpaced( N, delta/2, Rmax );
  rb.setLinSpaced( N, 0, Rmax-delta/2 );

  // std::cout << rb << std::endl;
  
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
  // ColPivHouseholderQR<Matrix3f> dec(A);
  // Vector3f x = dec.solve(b);
  std::cout << A << std::endl;
}

