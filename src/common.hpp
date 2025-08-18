// Get Lame's first parameter assuming homogeneous isotropic 
inline double GetLambda(double E, double nu)
{
    return E*nu/((1+nu)*(1-2*nu));
};

// Get Lame's second parameter assuming homogeneous isotropic 
inline double GetMu(double E, double nu)
{
    return (3.0/2.0)*(E/(2*(1-2*nu)) - GetLambda(E,nu));
};
