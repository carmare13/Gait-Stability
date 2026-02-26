#include "bayesH_core.hpp"
#include <cmath>
#include <cfloat>
#include <cstdlib>

// -----------------------------
// Declarations (same as original)
// -----------------------------
double accrej(double* data, double maxln, double add, double minu, double maxu, int n);
double maxphix(double* data, int n);
double fmin(double* data, double ax, double bx, int n);
double lnphix(double h, double* data, int n);
arma::vec acfhk(double h, double maxlag);
arma::vec ltza(arma::vec rr, double* xx, int n);
int levdur(double* r, int n, double* x, double* y, double* e, double EPS);
double levdet(int n, double* x);
double dot(int n, double* x, double* y);
double revdot(int n, double* x, double* y);

typedef double* VECTOR;
VECTOR Malloc(long n);
void Dalloc(VECTOR p);

// -----------------------------
// Public entry point (Python calls this)
// -----------------------------
arma::vec bayesH_samples_core(const arma::vec& data, unsigned int n_iter, unsigned int seed) {
    // In original code: randu() from Armadillo RNG.
    // We seed it for reproducibility.
    arma::arma_rng::set_seed(seed);

    double add  = 0.001;
    double minu = 0.001;
    double maxu = 0.999;

    int nn = (int)data.n_elem;

    VECTOR datax = Malloc(nn);
    for (int i = 0; i < nn; ++i) {
        datax[i] = data[(arma::uword)i];
    }

    double maxln = maxphix(datax, nn);
    arma::vec pdf = arma::zeros<arma::vec>(n_iter);

    for (unsigned int i = 0; i < n_iter; i++) {
        pdf((arma::uword)i) = accrej(datax, maxln, add, minu, maxu, nn);
    }

    Dalloc(datax);
    return pdf;
}

// -----------------------------
// Original functions (unchanged logic)
// -----------------------------

double accrej(double* data, double maxln, double add, double minu, double maxu, int n) {
    double dist = maxu - minu;
    double maxln1 = maxln - std::log(dist) + add;

    double u = arma::randu();
    double logu = std::log(u) + maxln1;

    double y = arma::randu(arma::distr_param(minu, maxu));

    while (logu > (lnphix(y, data, n) - std::log(dist))) {
        u = arma::randu();
        logu = std::log(u) + maxln1;
        y = arma::randu(arma::distr_param(minu, maxu));
    }

    return y;
}

double maxphix(double* data, int n) {
    double hmin = fmin(data, 0.00001, 0.99999, n);
    double maxln = lnphix(hmin, data, n);
    return maxln;
}

double fmin(double* data, double ax, double bx, int n) {
    const double c = (3 - std::sqrt(5.0)) * 0.5;
    double tol = std::pow(DBL_EPSILON, 0.25);
    double u;

    double eps = DBL_EPSILON;
    double tol1 = eps + 1;
    eps = std::sqrt(eps);

    double a = ax;
    double b = bx;
    double v = a + c * (b - a);
    double w = v;
    double x = v;
    double d = 0;
    double e = 0;

    double fx = lnphix(x, data, n);
    fx = -1 * fx;
    double fv = fx;
    double fw = fx;
    double tol3 = tol / 3;

    for (;;) {
        double xm = (a + b) * 0.5;
        double tol1_local = eps * std::fabs(x) + tol3;
        double t2 = tol1_local * 2;

        if (std::fabs(x - xm) <= t2 - (b - a) * 0.5) {
            break;
        }

        double p = 0;
        double q = 0;
        double r = 0;

        if (std::fabs(e) > tol1_local) {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2;

            if (q > 0) p = -p;
            else q = -q;

            r = e;
            e = d;
        }

        if (std::fabs(p) >= std::fabs(q * 0.5 * r) ||
            p <= q * (a - x) || p >= q * (b - x)) {
            if (x < xm) e = b - x;
            else e = a - x;
            d = c * e;
        } else {
            d = p / q;
            u = x + d;
            if (u - a < t2 || b - u < t2) {
                d = tol1_local;
                if (x >= xm) d = -d;
            }
        }

        if (std::fabs(d) >= tol1_local) u = x + d;
        else if (d > 0) u = x + tol1_local;
        else u = x - tol1_local;

        double fu = lnphix(u, data, n);
        fu = -1 * fu;

        if (fu <= fx) {
            if (u < x) b = x;
            else a = x;
            v = w; w = x; x = u;
            fv = fw; fw = fx; fx = fu;
        } else {
            if (u < x) a = u;
            else b = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }

    return x;
}

double lnphix(double h, double* data, int n) {
    int maxlag = n - 1;
    arma::vec acf = acfhk(h, maxlag);
    arma::vec q = ltza(acf, data, n);
    double f = -0.5 * q(3)
             - 0.5 * (maxlag) * std::log(q(2) * q(0) - std::pow(q(1), 2))
             + (0.5 * n - 1) * std::log(q(2));
    return f;
}

arma::vec acfhk(double h, double maxlag) {
    unsigned int n = (unsigned int)maxlag + 1;
    double h2 = h * 2;
    arma::vec acf(n);
    acf(0) = 1;

    for (unsigned int i = 0; i < (unsigned int)maxlag; i++) {
        double k = (double)i + 1.0;
        acf(i + 1) = 0.5 * (std::pow((k + 1), h2) - 2 * std::pow(k, h2) + std::pow((k - 1), h2));
    }

    return acf;
}

arma::vec ltza(arma::vec rr, double* xr, int n) {
    double EPS = DBL_EPSILON;
    int _fault1;
    int n1 = n - 1;
    VECTOR r, y1, y2, e1, e2, e3;

    arma::vec y = arma::zeros<arma::vec>(4);

    r = Malloc(n);
    for (int i = 0; i < n; ++i) {
        r[i] = rr((arma::uword)i);
    }

    y1 = Malloc(n);
    y2 = Malloc(n);
    e1 = Malloc(n1);
    e2 = Malloc(n1);
    e3 = Malloc(n);

    for (int i = 0; i < n; ++i) {
        e3[i] = 1;
    }

    _fault1 = levdur(r, n, xr, y1, e1, EPS);
    (void)_fault1;

    arma::vec yy1 = arma::zeros<arma::vec>(n);
    for (int i = 0; i < n; ++i) {
        yy1((arma::uword)i) = y1[i];
    }

    _fault1 = levdur(r, n, e3, y2, e2, EPS);

    arma::vec yy2 = arma::zeros<arma::vec>(n);
    for (int i = 0; i < n; ++i) {
        yy2((arma::uword)i) = y2[i];
    }

    y(3) = levdet(n1, e2);
    double s1 = arma::accu(yy2);
    double s2 = arma::accu(yy1);
    double s3 = dot(n, xr, y1);

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

int levdur(double* r, int n, double* x, double* y, double* e, double EPS) {
    (void)EPS;

    VECTOR v, l, b, c;
    int n1 = n - 1;
    int i, j, k, m;

    v = Malloc(n1);
    l = Malloc(n1);
    b = Malloc(n);
    c = Malloc(n1);

    e[0] = 1.0 - r[1] * r[1];
    v[0] = -r[1];
    l[0] = x[1] - r[1] * x[0];
    b[0] = -r[1];
    b[1] = 1.0;
    y[0] = (x[0] - r[1] * x[1]) / e[0];
    y[1] = l[0] / e[0];

    for (i = 1; i < n1; i++) {
        v[i] = -dot(i + 1, r + 1, b) / e[i - 1];
        e[i] = e[i - 1] * (1 - v[i] * v[i]);
        l[i] = x[i + 1] - revdot(i + 1, r + 1, y);

        for (k = 0; k < i + 1; k++) {
            c[k] = b[i - k];
        }

        b[i + 1] = b[i];

        for (j = i; j > 0; j--) {
            b[j] = b[j - 1] + v[i] * c[j];
        }

        b[0] = v[i] * c[0];
        y[i + 1] = (l[i] / e[i]) * b[i + 1];

        for (m = i; m > -1; m--) {
            y[m] = y[m] + (l[i] / e[i]) * b[m];
        }
    }

    Dalloc(v);
    Dalloc(l);
    Dalloc(b);
    Dalloc(c);

    return 0;
}

double levdet(int n, double* x) {
    double d = 0.0;
    for (int i = 0; i < n; i++) {
        d += std::log(x[i]);
    }
    return d;
}

VECTOR Malloc(long n) {
    // Original had calloc(n, n*sizeof(double)) (over-allocation).
    // This keeps identical numeric results but uses correct memory sizing.
    VECTOR vector = (VECTOR)std::calloc((size_t)n, sizeof(double));
    return vector;
}

void Dalloc(VECTOR p) {
    std::free(p);
}

double dot(int n, double* x, double* y) {
    double d = 0.0;
    for (int i = 0; i < n; i++) d += x[i] * y[i];
    return d;
}

double revdot(int n, double* x, double* y) {
    double d = 0.0;
    for (int i = 0; i < n; i++) d += x[n - 1 - i] * y[i];
    return d;
}