/**
 * This miniapp solves the Kirchoff-Love plate equation for clamped
 * boundary conditions using the C0 interior penalty method outlined in:
 *
 * Brenner, Susanne & Sung, Li-yeng. (2005). C0 Interior Penalty Methods
 * for Fourth Order Elliptic Boundary Value Problems on Polygonal Domains.
 * Journal of Scientific Computing. 22-23. 83-118. 10.1007/s10915-004-4135-7.
 */

#include <mfem.hpp>

using namespace mfem;
using namespace std;

struct KL_Context
{
   double Lx = 1.0;
   double Ly = 1.0;
   double t = 0.1;

   int Nx = 5;
   int Ny = 5;

   int rs = 0;
   int order = 2;

   // Material properties for 17-4PH stainless steel
   double E = 1.0;//196.5e9;
   double nu = 0.27;

   double delta_p_uniform = 0.1;//1e6;


   // Penalty coefficient
   double eta = 1e5;
} ctx;

class BiharmonicIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient &D;

   inline static const Vector factors_2D{1.0, 2.0, 1.0};
   mutable DenseMatrix hessian;
   mutable Vector factors;
public:
   BiharmonicIntegrator(Coefficient &D_) : D(D_) {}

   void AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat) override;
};

class C0InteriorPenaltyIntegrator : public BilinearFormIntegrator
{
   const double eta;

   // AssembleBlock Helpers:
   mutable Vector n_b, dnshape_a, dnshape_b, nd2nshape_b, nv;

   // AssembleFaceMatrix Helpers:
   mutable Vector normal_1, normal_2;
   mutable DenseMatrix dshape_1, dshape_2, hessian_1, hessian_2, block11, block12, block21, block22, elmat_p;

public:
   C0InteriorPenaltyIntegrator(double eta_) : eta(eta_) {};

   void AssembleBlock(const DenseMatrix &dshape_a, const DenseMatrix &dshape_b, const DenseMatrix &hessian_b, const Vector &n_a, const Vector &n_b, double h_e, DenseMatrix &elmat_ab);

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, DenseMatrix &elmat) override;
};

double dp_test(const Vector &x)
{
   double c_x = cos(2*M_PI*x[0]);
   double c_y = cos(2*M_PI*x[1]);
   return 4*c_x*c_y - c_x - c_y;
}

double w_exact(const Vector &x)
{
   return (1.0/(16.0*pow(M_PI,4)))*(cos(2*M_PI*x[0])-1)*(cos(2*M_PI*x[1])-1);
}

void grad_w_exact(const Vector &x, Vector &grad_w)
{
   grad_w.SetSize(2);
   grad_w[0] = (1.0/(16.0*pow(M_PI,4)))*(-2*M_PI*sin(2*M_PI*x[0]))*(cos(2*M_PI*x[1])-1);
   grad_w[1] = (1.0/(16.0*pow(M_PI,4)))*(-2*M_PI*sin(2*M_PI*x[1]))*(cos(2*M_PI*x[0])-1);
}

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   // Parse command line args
   OptionsParser args(argc, argv);
   args.AddOption(&ctx.Lx, "-Lx", "--length-x", "Panel length in x-direction.");
   args.AddOption(&ctx.Ly, "-Ly", "--length-y", "Panel length in y-direction.");
   args.AddOption(&ctx.t, "-t", "--thickness", "Panel thickness.");

   args.AddOption(&ctx.Nx, "-Nx", "--numelems-x", "Number of elements in panel x-direction.");
   args.AddOption(&ctx.Ny, "-Ny", "--numelems-y", "Number of elements in panel y-direction.");

   args.AddOption(&ctx.rs, "-rs", "--refine-serial", "Number of times to refine mesh in serial.");
   args.AddOption(&ctx.order, "-o", "--order", "Order of the finite elements.");

   args.AddOption(&ctx.E, "-E", "--youngs-modulus", "Young's modulus of the panel.");
   args.AddOption(&ctx.nu, "-nu", "--poisson-ratio", "Poisson ratio of the panel.");

   args.AddOption(&ctx.delta_p_uniform, "-dp", "--delta-p", "Uniform pressure difference imposed onto panel.");

   args.AddOption(&ctx.eta, "-eta", "--penalty-coeff", "Penalty coefficient.");


   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (rank == 0)
   {
      args.PrintOptions(cout);
   }

   // Generate panel mesh
   Mesh m = Mesh::MakeCartesian2D(ctx.Nx, ctx.Ny, Element::Type::QUADRILATERAL, true, ctx.Lx, ctx.Ly);
   m.SetCurvature(ctx.order); // ensure isoparametric!
   int dim = m.Dimension();

   // Refine the mesh
   for (int i = 0; i < ctx.rs; i++)
   {
      m.UniformRefinement();
   }

   // Partition the mesh
   ParMesh pmesh(MPI_COMM_WORLD, m);

   // Initialize the FE collection and FiniteElementSpace
   H1_FECollection fe_coll(ctx.order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fe_coll, 1);

   // Get the degrees-of-freedom (DOFs) associated with the sides of the panel
   // (these will be fixed/clamped)
   Array<int> all_bdr_marker(pmesh.bdr_attributes.Size());
   all_bdr_marker = 1; // Mark all sides

   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(all_bdr_marker, ess_tdof_list);
   // ess_tdof_list now holds indices of the DOFs associated with the panel sides

   // Create a coefficient representing the uniform pressure BC
   //ConstantCoefficient p_load(-ctx.delta_p_uniform);
   FunctionCoefficient p_load(dp_test);

   // Compute bending stiffness D using E, nu, and t, as coefficient
   double D_val = 1.0;//2*pow(ctx.t,3)*ctx.E/(3*(1-ctx.nu));
   ConstantCoefficient D(D_val);

   // Initialize the bilinear form representing the LHS (stiffness matrix K in Ku=f)
   ParBilinearForm k(&fespace);
   k.AddDomainIntegrator(new BiharmonicIntegrator(D));
   k.AddInteriorFaceIntegrator(new C0InteriorPenaltyIntegrator(ctx.eta));
   k.AddBdrFaceIntegrator(new C0InteriorPenaltyIntegrator(ctx.eta));

   // Initialize the linear form representing the LHS (forcing term f in Ku=f)
   ParLinearForm f(&fespace);
   f.AddDomainIntegrator(new DomainLFIntegrator(p_load));

   // Initialize the solution
   ParGridFunction W_gf(&fespace);
   W_gf = 0.0;
   W_gf.SetTrueVector();

   // Assemble the stiffness matrix
   k.Assemble();
   k.Finalize();
   std::unique_ptr<HypreParMatrix> K_mat(k.ParallelAssemble());

   // Assemble the RHS
   Vector F_vec(fespace.GetTrueVSize());
   f.Assemble();
   f.ParallelAssemble(F_vec);

   // Apply boundary conditions to the matrix system
   std::unique_ptr<HypreParMatrix> K_e(K_mat->EliminateRowsCols(ess_tdof_list));
   K_mat->EliminateBC(*K_e, ess_tdof_list, W_gf.GetTrueVector(), F_vec);

   // Initialize preconditioner
   HypreBoomerAMG amg(*K_mat);

   // Initialize solver
   HyprePCG pcg(*K_mat);
   pcg.SetTol(1e-8);
   pcg.SetMaxIter(100000);
   pcg.SetPrintLevel(2);
   pcg.SetPreconditioner(amg);
   
   // Solve Ku=f
   pcg.Mult(F_vec, W_gf.GetTrueVector());
   W_gf.SetFromTrueVector();

   // // Write the output
   ParaViewDataCollection pvdc("KirchoffLove", &pmesh);
   pvdc.SetHighOrderOutput(true);
   pvdc.RegisterField("Deformation", &W_gf);

   FunctionCoefficient exact_coeff(w_exact);
   VectorFunctionCoefficient exact_grad_coeff(2, grad_w_exact);

   ParGridFunction exact_gf(&fespace);
   exact_gf.ProjectCoefficient(exact_coeff);
   pvdc.RegisterField("Exact", &exact_gf);
   pvdc.Save();

   double err_L2 = W_gf.ComputeL2Error(exact_coeff);
   double err_H1 = W_gf.ComputeH1Error(&exact_coeff, &exact_grad_coeff);
   if (rank == 0)
   {
      cout << "L2 Error: " << err_L2 << endl;
      cout << "H1 Error: " << err_H1 << endl;
   }

   return 0;
}


void BiharmonicIntegrator::AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int ndof = el.GetDof();
   int dim = el.GetDim();

   MFEM_ASSERT(dim == 2, "Dimension must be 2.");

   double c, w;

   hessian.SetSize(ndof, dim * (dim + 1) / 2);
   elmat.SetSize(ndof);
   factors.SetSize(dim * (dim + 1) / 2);

   elmat = 0.0;

   const IntegrationRule *ir = GetIntegrationRule(el, Trans);
   if (ir == NULL)
   {
      int order = 2*el.GetOrder() + Trans.OrderW();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);

      el.CalcPhysHessian(Trans, hessian);
      
      factors = factors_2D;
      factors *= D.Eval(Trans, ip) * ip.weight * Trans.Weight();

      AddMultADAt(hessian, factors, elmat);
   }
}

void C0InteriorPenaltyIntegrator::AssembleBlock(const DenseMatrix &dshape_a, const DenseMatrix &dshape_b, const DenseMatrix &hessian_b, const Vector &n_a, const Vector &n_b, double h_e, DenseMatrix &elmat_ab)
{  
   elmat_ab = 0.0;

   dnshape_a.SetSize(dshape_a.NumRows()); // ndofs_a
   dnshape_b.SetSize(dshape_b.NumRows()); // ndofs_b
   nd2nshape_b.SetSize(hessian_b.NumRows()); // ndofs_b
   nv.SetSize(hessian_b.NumCols());

   // dshape_a = (dof, dim)
   dshape_a.Mult(n_a, dnshape_a);

   // dshape_b = (dof, dim)
   dshape_b.Mult(n_b, dnshape_b);

   // hessian_b = (dof, [3 in 2D])
   nv[0] = pow(n_b[0],2);
   nv[1] = 2*n_b[0]*n_b[1];
   nv[2] = n_b[1]*n_b[1];

   hessian_b.Mult(nv, nd2nshape_b);

   // Consistency term:
   AddMult_a_VWt(1.0, dnshape_a, nd2nshape_b, elmat_ab);

   // Penalty term (symmetric):
   AddMult_a_VWt(eta/h_e, dnshape_a, dnshape_b, elmat_ab);
}

/** Compute: q^(a,b) + p^(a,b)
         
         q^(a,b) = [d\phi^(a)/dn^(a)][d^2\phi^b/dn^(a)^2], or
         q^(a,b) = [(grad \phi^a) dot n^(a)]*[n^(b)^T dot hess(\phi)^b dot n^(b)]

         and

         p^(a,b) = [d\phi^(a)/dn^(a)][d\phi^(b)/dn^(b)] or
                 = [(grad \phi^a) dot n^(a)][(grad \phi^b) dot n^(b)]
*/
void C0InteriorPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int dim = el1.GetDim();
   int ndof1 = el1.GetDof();
   int ndof2 = 0;
   MFEM_ASSERT(dim == 2, "Dimension must be 2.");

   // For boundary face integration:
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
   }

   normal_1.SetSize(dim);
   dshape_1.SetSize(ndof1, dim);
   hessian_1.SetSize(ndof1, dim * (dim + 1) / 2);
   block11.SetSize(ndof1, ndof1);
   if (ndof2 > 0)
   {
      dshape_2.SetSize(ndof2, dim);
      normal_2.SetSize(dim);
      hessian_2.SetSize(ndof2, dim * (dim + 1) / 2);
      block12.SetSize(ndof1, ndof2);
      block21.SetSize(ndof2, ndof1);
      block22.SetSize(ndof2, ndof2);
   }
   elmat.SetSize(ndof1 + ndof2);
   elmat_p.SetSize(ndof1 + ndof2);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * max(el1.GetOrder(), ndof2 ? el2.GetOrder() : 0);
      ir = &IntRules.Get(el1.GetGeomType(), order);
   }


   // Compute edge length
   double h_e  = 0.0;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetAllIntPoints(&ip);
      h_e += ip.weight * Trans.Face->Weight();
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Compute normals, derivatives, and hessians
      CalcOrtho(Trans.Face->Jacobian(), normal_1);
      normal_1 /= normal_1.Norml2();
      el1.CalcPhysDShape(*Trans.Elem1, dshape_1);
      el1.CalcPhysHessian(*Trans.Elem1, hessian_1);
      if (ndof2)
      {
         normal_2 = normal_1;
         normal_2 *= -1.0;
         el2.CalcPhysDShape(*Trans.Elem2, dshape_2);
         el2.CalcPhysHessian(*Trans.Elem2, hessian_2);
      }

      // (1,1) block
      AssembleBlock(dshape_1, dshape_1, hessian_1, normal_1, normal_1, h_e, block11);
      elmat_p.SetSubMatrix(0, 0, block11);
      if (ndof2 > 0)
      {
         // (1,2) block
         AssembleBlock(dshape_1, dshape_2, hessian_2, normal_1, normal_2, h_e, block12);
         elmat_p.SetSubMatrix(0, ndof1, block12);

         // (2,1) block
         AssembleBlock(dshape_2, dshape_1, hessian_1, normal_2, normal_1, h_e, block21);
         elmat_p.SetSubMatrix(ndof1, 0, block21);

         // (2,2) block
         AssembleBlock(dshape_2, dshape_2, hessian_2, normal_2, normal_2, h_e, block22);
         elmat_p.SetSubMatrix(ndof1, ndof1, block22);
      }

      // Apply 1/2 factor and symmetry term
      elmat_p.Symmetrize();

      elmat_p *= ip.weight * Trans.Face->Weight();

      elmat += elmat_p;
   }
}