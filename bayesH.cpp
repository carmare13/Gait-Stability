/* pdf = bayesH(x, n)
 * inputs  - x, A real values vector (i.e., time series data) to be
 *              analyzed.
 *		   - n, The number of iterations to sample the posterior
 *              distribution of Hurst exponent
 *
 * outputs - pdf, Estimated probability density function of Hurst exponent
 *
 * author  - Seung Kyeom Kim, 2023.
 *
 * remarks - Returns probability density function of Hurst exponent.
 *         - The median of the probability density function can be used as
 *           the estimated Hurst exponent.
 * reference
 *  Tyralis, H., & Koutsoyiannis, D. (2014). A Bayesian statistical model for deriving the predictive distribution of hydroclimatic variables. Climate dynamics, 42, 2867-2883.
 *  Likens, A. D., Mangalam, M., Wong, A. Y., Charles, A. C., & Mills, C. (2023). Better than DFA? A Bayesian method for estimating the Hurst exponent in behavioral sciences. arXiv preprint arXiv:2301.11262.
 *
 */

#include <mex.h>
#include "armaMex.hpp"

/* Sampling related functions */
double accrej(double* data, double maxln, double add, double minu, double maxu, int n);
/* Posterior distribution related functions */
double maxphix(double* data, int n);
double fmin(double* data, double ax, double bx, int n);
double lnphix(double h, double* data, int n);
arma::vec acfhk(double h, double maxlag);
arma::vec ltza(arma::vec rr, double* xx, int n);
/* Levinson-Durbin algorithm related functions */
int levdur(double* r, int n, double* x, double* y, double* e, double EPS);
double levdet(int n, double* x);
/* Algebraic functions */
double dot(int n, double* x, double* y);
double revdot(int n, double* x, double* y);
/* Pointer related functions */
typedef double* VECTOR;
VECTOR Malloc( long n );
void Dalloc( VECTOR Malloc );

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    
    arma::vec data = armaGetPr(prhs[0]);
    unsigned int n = armaGetScalar<unsigned int>(prhs[1]);
    
    double add = 0.001;
    double minu = 0.001;
    double maxu = 0.999;
    
    int nn = data.n_elem;
    
    VECTOR datax;
    datax = Malloc(nn);
    for (unsigned int i = 0; i < nn; ++i){
        datax[i] = data[i];
    }
    
    double maxln = maxphix(datax, nn);
    arma::vec pdf = zeros(n);

    for (unsigned int i = 0; i < n; i ++)
    {
        pdf(i) = accrej(datax, maxln, add, minu, maxu, nn);
    }
    
    Dalloc(datax);
    
    plhs[0] = armaCreateMxMatrix(1, n, mxDOUBLE_CLASS);
    
    armaSetData(plhs[0], pdf);
    
}

/* Sample Hurst exponent based on accep-reject algorithm */
double accrej(double* data, double maxln, double add, double minu, double maxu, int n) {
    
    double dist = maxu - minu;
    double maxln1 = maxln - log(dist) + add;
    
    double u = randu();
    double logu = log(u) + maxln1;
    
    double y = randu(distr_param(minu,maxu));
    
    while (logu > (lnphix(y,data, n) - log(dist))) {
        
        u = randu();
        
        logu = log(u) + maxln1;
        
        y = randu(distr_param(minu,maxu));
        
    }
    
    return y;
    
}

/* Return the maximum value of lnphix */
double maxphix(double* data, int n) {
    
    double hmin = fmin(data, 0.00001, 0.99999, n);
    double maxln = lnphix(hmin, data, n);
    
    return maxln;
    
}

/* Return the value that minimizes the outcome of given function based on
 * Brent's method */
double fmin(double* data, double ax, double bx, int n) {
    
    const double c = (3 - sqrt(5)) * 0.5;
    double tol = pow(DBL_EPSILON,0.25);
    double u;
    double eps = DBL_EPSILON;
    double tol1 = eps + 1;
    eps = sqrt(eps);
    double a = ax;
    double b = bx;
    double v = a + c * (b - a);
    double w = v;
    double x = v;
    double d = 0;
    double e = 0;
    double fx = lnphix(x, data, n);
    fx = -1*fx;
    double fv = fx;
    double fw = fx;
    double tol3 = tol / 3;
    
    for(;;) {
        double xm = (a + b) * 0.5;
        double tol1 = eps * fabs(x) + tol3;
        double t2 = tol1 * 2;
        if (fabs(x - xm) <= t2 - (b - a) * 0.5) {
            break;
        }
        double p = 0;
        double q = 0;
        double r = 0;
        if (fabs(e) > tol1) {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2;
            if (q > 0) {
                p = -p;
            }
            else {
                q = -q;
            }
            r = e;
            e = d;
        }
        if (fabs(p) >= fabs(q * 0.5 * r) ||
                p <= q * (a - x) || p >= q * (b - x)) {
            if (x < xm) {
                e = b - x;
            }
            else {
                e = a - x;
            }
            d = c * e;
        }
        else {
            d = p / q;
            u = x + d;
            if (u - a < t2 || b - u < t2) {
                d = tol1;
                if (x >= xm) {
                    d = -d;
                }
            }
        }
        if (fabs(d) >= tol1) {
            u = x + d;
        }
        else if (d > 0) {
            u = x + tol1;
        }
        else {
            u = x - tol1;
        }
        double fu = lnphix(u, data, n);
        fu = -1*fu;
        if (fu <= fx) {
            if (u < x) {
                b = x;
            }
            else {
                a = x;
            }
            v = w;    w = x;   x = u;
            fv = fw; fw = fx; fx = fu;
        } else {
            if (u < x) {
                a = u;
            }
            else {
                b = u;
            }
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            }
            else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }
    
    return x;
}

/* Compute natural logarithm of Eq.10 from Tyralis & Koutsoyiannis (2014) */
double lnphix(double h, double* data, int n) {
    
    int maxlag = n - 1;
    arma::vec acf = acfhk(h, maxlag);
    arma::vec q = ltza(acf, data, n);
    double f = -0.5*q(3) - 0.5*(maxlag) * log(q(2)*q(0)-pow(q(1),2)) + (0.5*n - 1)*log(q(2));
    
    return f;
    
}

/* Generate autocorrelation function for Hurst-Kolmogorov (HK) process
 * based on desired Hurst exponent */
arma::vec acfhk(double h, double maxlag) {
    
    
    unsigned int n = maxlag + 1;
    double h2 = h*2;
    arma::vec acf(n);
    acf(0) = 1;
    
    for (unsigned int i=0; i < maxlag; i++) {
        double k = i+1;
        acf(i+1) = 0.5*(pow((k+1),h2) - 2*pow(k,h2) + pow((k-1),h2));
    }
    
    return acf;
    
}

/* Compute quadratic forms for the inverse of a symmetric, positive definite
 * autocorrelation matrix */
arma::vec ltza(arma::vec rr, double* xr, int n) {
    
    double EPS = DBL_EPSILON;
    int _fault1;
    int n1 = n-1;
    VECTOR r, y1, y2, e1, e2, e3;
    
    arma::vec y = zeros<arma::vec>(4);
    
    r = Malloc(n);
    for (unsigned int i = 0; i < n; ++i){
        r[i] = rr[i];
    }
    
    y1 = Malloc(n);
    y2 = Malloc(n);
    e1 = Malloc(n1);
    e2 = Malloc(n1);
    e3 = Malloc(n);
    for (unsigned int i = 0; i < n; ++i){
        e3[i] = 1;
    }
    
    _fault1 = levdur(r,n,xr,y1,e1,EPS);
    arma::vec yy1 = zeros<arma::vec>(n);
    for (int i = 0; i < n; ++i){
        yy1[i] = y1[i];
    }
    
    _fault1 = levdur(r,n,e3,y2,e2,EPS);
    arma::vec yy2 = zeros<arma::vec>(n);
    for (int i = 0; i < n; ++i){
        yy2[i] = y2[i];
    }
    
    y(3) = levdet(n1,e2);
    double s1 = arma::accu(yy2);
    double s2 = arma::accu(yy1);
    double s3 = dot(n,xr,y1);
    y(0) = s3;
    y(1) = s2;
    y(2) = s1;
    
    Dalloc(r);
    Dalloc(y1);
    Dalloc(y2);
    Dalloc(e1);
    Dalloc(e2);
    Dalloc(e3);
    
    return y;
}

/* Levinson-Durbin recursion */
int levdur(double* r, int n, double* x, double* y, double* e, double EPS){
    
    VECTOR v, l, b, c;
    
    int n1;
    int i, j, k, m;
    n1 = n-1;
    
    v = Malloc(n1);
    l = Malloc(n1);
    b = Malloc(n);
    c = Malloc(n1);
    
    e[0] = 1.0-r[1]*r[1];
    v[0] = -r[1];
    l[0] = x[1]-r[1]*x[0];
    b[0] = -r[1];
    b[1] = 1.0;
    y[0] = (x[0]-r[1]*x[1])/e[0];
    y[1] = l[0]/e[0];
    
    for (i = 1; i < n1; i++){
        v[i] = -dot(i+1, r+1, b)/e[i-1];
        e[i] = e[i-1]*(1-v[i]*v[i]);
        l[i] = x[i+1]-revdot(i+1, r+1, y);
        
        for (k = 0; k < i + 1; k++){
            c[k] = b[i-k];
        }
        b[i+1] = b[i];
        
        for (j = i; j > 0; j--){
            b[j] = b[j-1]+v[i]*c[j];
        }
        
        b[0] = v[i] * c[0];
        y[i+1] = (l[i]/e[i])*b[i+1];
        
        for (m = i; m > -1; m--){
            y[m] = y[m]+(l[i]/e[i])*b[m];
        }
    }
    
    Dalloc(v);
    Dalloc(l);
    Dalloc(b);
    Dalloc(c);
    
    return 0;
    
}

/* Compute determinant with Levinson algorithm */
double levdet(int n, double* x)
{
    int i;
    double d;
    d = 0.0;
    for (i = 0; i < n; i++){
        d += log(x[i]);
    }
    return d;
}

/* Allocate memory of a vector and create pointer variable */
VECTOR Malloc( long n )
{
    
    
    VECTOR vector;
    vector=(VECTOR)calloc(n, n*sizeof(double));
    
    return vector;
}

/* Deallocates memory of a vector */
void Dalloc( VECTOR Malloc )
{
    free( Malloc );
}

/* Dot product of two vectors */
double dot(int n,double* x,double* y)
{
    double d = 0.0;
    int i;
    for (i = 0; i < n; i++)
        d += x[i]*y[i];
    return d;
}

/* Dot product of two vectors (but with one of the vectors reversed) */
double revdot(int n,double* x,double* y)
{
    double d = 0.0;
    int i;
    for (i = 0; i < n; i++)
        d += x[n-1-i]*y[i];
    return d;
}

