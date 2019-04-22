#define _USE_MATH_DEFINES

#include <cmath>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <boost/math/special_functions/bessel.hpp> 
using namespace Eigen;

int N1 = 50;
int N2 = 50; // If 1-Region N2=0 
int Nt = N1 + N2;

// ------ Define Geometry ----------
double R1 = 1; // If R1=0 one region
double R2 = 1;
bool region = true; // If region=true two regions, if region=false one region
bool reflected = false; // If reflected=true reflected boundary condition used

// ------ Nuclear Data ----------------
Vector2d D( 0.834189716667413, 0.916060403234839 );
Vector2d Sig_a( 0.0695012, 0.0003208 );
Vector2d nuSig_f( 0.158502888, 0 );

auto get_rc = [] ( auto& R1, auto& R2, auto& N1, auto& N2, auto& reg ) {
		int Nt = N1 + N2;
		VectorXd rc(Nt);
		rc.setZero();  
		if ( !reg ) {
		  rc.setLinSpaced( N1, R2/(2*N1), R2 );
		} else {
		  rc.head(N1).setLinSpaced( N1, 0 + R1/(2*N1), R1 );
		  rc.tail(N2).setLinSpaced( N2, R1+(R2/N2), R1+R2 );
		}
		return rc;
	      };

auto get_rb = [] ( auto& R1, auto& R2, auto& N1, auto& N2, auto& reg ) {
		int Nt = N1 + N2;
		VectorXd rb(Nt);
		rb.setZero();  
		if ( !reg ) {
		  rb.setLinSpaced( N1, R1, R2-R2/(2*N1) );
		} else {
		  rb.head(N1).setLinSpaced( N1, 0, R1-R1/(2*N1) );
		  rb.tail(N2).setLinSpaced( N2, R1+R2/(N2*2), R1+R2-R2/(N2*2) );
		}
		return rb;
	      };

auto get_A11 = [&] ( auto& N, auto& R1, auto& R2, Vector2d D, Vector2d Sig_a, auto& refl ) {
		 VectorXd rb(N); VectorXd rc(N);
		 rc.setLinSpaced( N, R2/(2*N), R2 );
		 rb.setLinSpaced( N, R1, R2-R2/(2*N) );
		 
		 MatrixXd A(N-1,N-1);
		 
		 A(0,0) = D(0)*rb(1)/( rc(0) * ( rc(1) - rc(0) ) * ( rb(1) - rb(0) ) ) + Sig_a(0);
		 A(0,1) = (-1)*D(0)*rb(1)/( rc(0) * ( rc(1) - rc(0) ) * ( rb(1) - rb(0) ) );
		 if (refl) {
		   A(N-2,N-2) = D(0)*rb(N-1)/( rc(N-2) * ( rc(N-1) - rc(N-2) ) * ( rb(N-1) - rb(N-2) ) ) + Sig_a(0)
		     + D(0)*rb(N-2)/( rc(N-2) * ( rc(N-2) - rc(N-3) ) * ( rb(N-2) - rb(N-3) ) )
		     - D(0)*rb(N-1)/( rc(N-2) * ( rc(N-1) - rc(N-2) ) * ( rb(N-1) - rb(N-2) ) );
		 } else {
		   A(N-2,N-2) = D(0)*rb(N-1)/( rc(N-2) * ( rc(N-1) - rc(N-2) ) * ( rb(N-1) - rb(N-2) ) ) + Sig_a(0)
		     + D(0)*rb(N-2)/( rc(N-2) * ( rc(N-2) - rc(N-3) ) * ( rb(N-2) - rb(N-3) ) );
		 }
		 A(N-2,N-3) = (-1)*D(0)*rb(N-2)/( rc(N-2) * ( rc(N-2) - rc(N-3) ) * ( rb(N-2) - rb(N-3) ) );
		 
		 for( int i=1; i<N-2; ++i ){
		   A(i,i) = D(0)*rb(i+1)/( rc(i) * ( rc(i+1) - rc(i) ) * ( rb(i+1) - rb(i) ) ) + Sig_a(0)
		     + D(0)*rb(i)/( rc(i) * ( rc(i) - rc(i-1) ) * ( rb(i) - rb(i-1) ) ); 
		   A(i,i+1) = (-1)*D(0)*rb(i+1)/( rc(i) * ( rc(i+1) - rc(i) ) * ( rb(i+1) - rb(i) ) );
		   A(i,i-1) = (-1)*D(0)*rb(i)/( rc(i) * ( rc(i) - rc(i-1) ) * ( rb(i) - rb(i-1) ) );
		 }
		 
		 return A;
	       };

auto get_A12 = [&] ( auto& N1, auto& N2, auto& R1, auto& R2, Vector2d D, Vector2d Sig_a, auto& refl ) {
		 int Nt = N1 + N2;
		 VectorXd rb(Nt); VectorXd rc(Nt);
		 rb.setZero(); rc.setZero();

		 rb.head(N1).setLinSpaced( N1, 0, R1-R1/(2*N1) );
		 rb.tail(N2).setLinSpaced( N2, R1+R2/(N2*2), R1+R2-R2/(N2*2) );
		 rc.head(N1).setLinSpaced( N1, rb(1) - R1/(2*N1), R1 );
		 rc.tail(N2).setLinSpaced( N2, R1+(R2/N2), R1+R2 );

		 MatrixXd A(Nt-1,Nt-1);
		 double dv1 = M_PI*(pow( rc(N1-1), 2 ) - pow( rb(N1-1), 2 ))/2;
		 double dv2 = M_PI*(pow( rb(N1), 2 ) - pow( rc(N1-1), 2 ))/2;
		 
		 A(0,0) = ( D(0)*rb(1))/((rc(0)*(rb(1)-rb(0))*(rc(1)-rc(0)))) + Sig_a(0);
		 A(0,1) = (-D(0)*rb(1))/((rc(0)*(rb(1)-rb(0))*(rc(1)-rc(0))));

		 if (refl) {
		   A(Nt-2,Nt-2) = D(1)*rb(Nt-1)/( rc(Nt-2) * ( rc(Nt-1) - rc(Nt-2) ) * ( rb(Nt-1) - rb(Nt-2) ) )
		     + Sig_a(1)
		     + D(1)*rb(Nt-2)/( rc(Nt-2) * ( rc(Nt-2) - rc(Nt-3) ) * ( rb(Nt-2) - rb(Nt-3) ) )
		     - D(1)*rb(Nt-1)/( rc(Nt-2) * ( rc(Nt-1) - rc(Nt-2) ) * ( rb(Nt-1) - rb(Nt-2) ) );
		 } else {
		   A(Nt-2,Nt-2) = D(1)*rb(Nt-1)/( rc(Nt-2) * ( rc(Nt-1) - rc(Nt-2) ) * ( rb(Nt-1) - rb(Nt-2) ) )
		     + Sig_a(1)
		     + D(1)*rb(Nt-2)/( rc(Nt-2) * ( rc(Nt-2) - rc(Nt-3) ) * ( rb(Nt-2) - rb(Nt-3) ) );
		 }
		   
		 A(Nt-2,Nt-3) = (-1)*D(1)*rb(Nt-2)/( rc(Nt-2) * (rc(Nt-2) - rc(Nt-3)) * (rb(Nt-2) - rb(Nt-3)) );

		 for( int i=1; i<Nt-2; ++i ){
		   if ( rc(i) < R1 ) { 
		     A(i,i) = D(0)*rb(i+1)/( rc(i) * ( rc(i+1) - rc(i) ) * ( rb(i+1) - rb(i) ) )
		       + Sig_a(0) + D(0)*rb(i)/( rc(i) * ( rc(i) - rc(i-1) ) * ( rb(i) - rb(i-1) ) ); 
		     A(i,i+1) = (-1)*D(0)*rb(i+1)/( rc(i) * ( rc(i+1) - rc(i) ) * ( rb(i+1) - rb(i) ) );
		     A(i,i-1) = (-1)*D(0)*rb(i)/( rc(i) * ( rc(i) - rc(i-1) ) * ( rb(i) - rb(i-1) ) );
		   } else if ( rc(i) == R1 ) {
		     A(i,i) = D(1)*rb(i+1)/( rc(i) * ( rc(i+1) - rc(i) ) * ( rb(i+1) - rb(i) ) )
		       + ( dv1*Sig_a(0) + dv2*Sig_a(1) ) / ( dv1 + dv2 ) 
		       + D(0)*rb(i)/( rc(i) * ( rc(i) - rc(i-1) ) * ( rb(i) - rb(i-1) ) ); 
		     A(i,i+1) = (-1)*D(1)*rb(i+1)/( rc(i) * ( rc(i+1) - rc(i) ) * ( rb(i+1) - rb(i) ) );
		     A(i,i-1) = (-1)*D(0)*rb(i)/( rc(i) * ( rc(i) - rc(i-1) ) * ( rb(i) - rb(i-1) ) );
		   }
		   else {
		     A(i,i) = D(1)*rb(i+1)/( rc(i) * ( rc(i+1) - rc(i) ) * ( rb(i+1) - rb(i) ) ) + Sig_a(1)
		       + D(1)*rb(i)/( rc(i) * ( rc(i) - rc(i-1) ) * ( rb(i) - rb(i-1) ) ); 
		     A(i,i+1) = (-1)*D(1)*rb(i+1)/( rc(i) * ( rc(i+1) - rc(i) ) * ( rb(i+1) - rb(i) ) );
		     A(i,i-1) = (-1)*D(1)*rb(i)/( rc(i) * ( rc(i) - rc(i-1) ) * ( rb(i) - rb(i-1) ) );
		   }
		 }
		 return A;
	       };

auto setup = [&] ( auto& N1, auto& N2, auto& R1, auto& R2, auto& reg, auto& refl, Vector2d D, Vector2d Sig_a ) {
	       MatrixXd A;
	       if (!reg) {
		 A = get_A11( N1, R1, R2, D, Sig_a, refl );
	       }
	       if (reg) {
		 A = get_A12( N1, N2, R1, R2, D, Sig_a, refl );
	       }
	       return A;
	     };

auto gen_b = [] ( VectorXd phi, VectorXd& b, VectorXd& rc, VectorXd& rb, Vector2d nuSig_f, auto& k, auto& N1,
		  auto& N2, auto& R1, auto& R2, auto& reg ) {
	       if ( !reg ) {
		 for( int i=0; i<N1-1; ++i ){
		   b(i) = (1/k)*nuSig_f(0)*phi(i);
		 }
	       }
	       if (reg) {
		 double dv1 = M_PI*(pow( rc(N1-1), 2 ) - pow( rb(N1-1), 2 ))/2;
		 double dv2 = M_PI*(pow( rb(N1), 2 ) - pow( rc(N1-1), 2 ))/2;
		 auto Nt = N1 + N2;
		 for( int i=0; i<Nt-1; ++i ){
		   if ( i < N1 ) {
		     b(i) = (1/k)*nuSig_f(0)*phi(i);
		   } else if ( i == N1 ) {
		     b(i) = (1/k)*(( nuSig_f(0)*dv1 + nuSig_f(1)*dv2 ) / (dv1+dv2))*phi(i);
		   } else {
		     b(i) = (1/k)*nuSig_f(1)*phi(i);
		   }
		 }
	       }
	     };

auto calc_k = [] ( VectorXd phi_n, VectorXd phi_o, VectorXd Rc, VectorXd Rb, Vector2d nuSig_f, auto k_o,
		   auto& N1, auto& N2, auto& reg ) {
		
		double numerator = 0.0;
		double denominator = 0.0;
		
		if (!reg) {
		  for( int i = 0; i<N1-1; ++i ){
		    numerator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )*nuSig_f(0)*phi_n(i);
		    denominator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )*nuSig_f(0)*phi_o(i);
		  }
		}
		if (reg) {
		double dv1 = M_PI*(pow( Rc(N1-1), 2 ) - pow( Rb(N1-1), 2 ))/2;
		double dv2 = M_PI*(pow( Rb(N1), 2 ) - pow( Rc(N1-1), 2 ))/2;

		  for( int i = 0; i<N1+N2-1; ++i ){
		    if ( Rc(i) < R1 ) {
		      numerator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )*nuSig_f(0)*phi_n(i);
		      denominator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )*nuSig_f(0)*phi_o(i);
		    } else if ( Rc(i) == R1 ) {
		      numerator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )
		      	*(( nuSig_f(0)*dv1 + nuSig_f(1)*dv2 )/(dv1+dv2))*phi_n(i);
		      denominator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )
			*(( nuSig_f(0)*dv1 + nuSig_f(1)*dv2 )/(dv1+dv2))*phi_o(i);
		    } else {
		      numerator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )*nuSig_f(1)*phi_n(i);
		      denominator += M_PI*( pow( Rc(i+1), 2 ) - pow( Rc(i), 2 ) )*nuSig_f(1)*phi_o(i);
		    }
		  }
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
  MatrixXd A  = setup( N1, N2, R1, R2, region, reflected, D, Sig_a );
  VectorXd Rc = get_rc( R1, R2, N1, N2, region );
  VectorXd Rb = get_rb( R1, R2, N1, N2, region );

  FullPivLU<Ref<MatrixXd> > lu(A);
  // // ColPivHouseholderQR<MatrixXd> dec(A);
  VectorXd phi = init_phi( Nt );
  VectorXd phi_n = init_phi( Nt );
  double k = 1.0;

  VectorXd b(Nt - 1);
  gen_b( phi, b, Rc, Rb, nuSig_f, k, N1, N2, R1, R2, region );


  
  for( int i = 0; i < 1000; ++i ){
    gen_b( phi, b, Rc, Rb, nuSig_f, k, N1, N2, R1, R2, region );
    phi_n = lu.solve(b);
    k = calc_k( phi_n, phi, Rc, Rb, nuSig_f, k, N1, N2, region );
    phi = phi_n;
  }
  std::cout << k << std::endl;

  printer( Rc, phi, Nt );
  
}
