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
   double t   = 0.1;

   int Nx  = 10;
   int Ny  = 10;

   int rs    = 0;
   int order = 2;

   // Material properties for 17-4PH stainless steel
   double E  = 196.5e9;
   double nu = 0.27;

   double delta_p_uniform = 1e3;

} ctx;


class BiharmonicIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient &D;
   
   mutable DenseMatrix d2shape;
public:
   BiharmonicIntegrator(Coefficient &D_) : D(D_) {}

   void AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat) override
   {
      int dof = el.GetDof();
      int dim = el.GetDim();
      double c, w;

      d2shape.SetSize(dof, dim);
      elmat.SetSize(dof, dof);

      const IntegrationRule *ir = GetIntegrationRule(el, Trans);
      elmat = 0.0;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      
         el.CalcHessian(ip, d2shape); // D^2\phi
      
         Trans.SetIntPoint(&ip);
         w = D.Eval(Trans, ip)*ip.weight*Trans.Weight(); // D * w_IP * |J_\tau|

         c = fcoeff.Eval(Tr,ip);
         w = c*ip.weight*Tr.Weight();
         mfem::Mult(dshape, Tr.InverseJacobian(), gshape);
         gshape.GradToDiv(pelvect);

         pelvect *= w;
         elvect += pelvect;
      }
   }
};

class ConsistencyIntegrator : public BilinearFormIntegrator
{

public:
   ConsistencyIntegrator();

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, DenseMatrix &elmat) override
   {

   }
};

class PenaltyIntegrator : public BilinearFormIntegrator
{

public:
   PenaltyIntegrator();

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, DenseMatrix &elmat) override
   {

   }
};


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
   ConstantCoefficient p_load(-ctx.delta_p_uniform);

   // Compute bending stiffness D using E, nu, and t, as coefficient
   double D_val = 
   ConstantCoefficient D(lambda_val);

   // Initialize the bilinear form representing the LHS (stiffness matrix K in Ku=f)
   ParBilinearForm k(&fespace);
   k.AddDomainIntegrator(new BiharmonicIntegrator());
   k.AddInteriorFaceIntegrator(new ConsistencyIntegrator());
   k.AddInteriorFaceIntegrator(new PenaltyIntegrator());

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

   // // Initialize preconditioner
   // HypreBoomerAMG amg(*K_mat);
   // amg.SetElasticityOptions(&fespace);

   // // Initialize solver
   // HyprePCG pcg(*K_mat);
   // pcg.SetTol(1e-8);
   // pcg.SetMaxIter(1000);
   // pcg.SetPrintLevel(2);
   // pcg.SetPreconditioner(amg);

   // // Solve Ku=f
   // pcg.Mult(F_vec, W_gf.GetTrueVector());
   // U_gf.SetFromTrueVector();

   // // Write the output
   // ParaViewDataCollection pvdc("KirchoffLove", &pmesh);
   // pvdc.RegisterField("Deformation", &W_gf);
   // pvdc.Save();

   return 0;
}