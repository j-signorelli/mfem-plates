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
   double eta = 10;

   bool quads = false;

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
private:
   const double eta;

   // AssembleFaceMatrix Helpers:
   mutable Vector normal[2], dnshape[2], nv[2], nd2nshape[2];
   mutable DenseMatrix dshape[2], hessian[2], blockJ[2][2], blockC[2][2], elmatJ_p, elmatC_p;
public:
   C0InteriorPenaltyIntegrator(double eta_) : eta(eta_) {};

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

   args.AddOption(&ctx.rs, "-rs", "--refinements", "Number of times to refine mesh.");
   args.AddOption(&ctx.order, "-o", "--order", "Order of the finite elements.");

   args.AddOption(&ctx.E, "-E", "--youngs-modulus", "Young's modulus of the panel.");
   args.AddOption(&ctx.nu, "-nu", "--poisson-ratio", "Poisson ratio of the panel.");

   args.AddOption(&ctx.delta_p_uniform, "-dp", "--delta-p", "Uniform pressure difference imposed onto panel.");

   args.AddOption(&ctx.eta, "-eta", "--penalty-coeff", "Penalty coefficient.");

   args.AddOption(&ctx.quads, "-qd", "--quads", "-ti", "--tris", "Use quads or tris.");

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
   Mesh m;
   if (ctx.quads)
   {
      m = Mesh::MakeCartesian2D(ctx.Nx, ctx.Ny, Element::Type::QUADRILATERAL, true, ctx.Lx, ctx.Ly);
   }
   else
   {
      m = Mesh("tri_plate.mesh", 1, 0);
   }
   m.SetCurvature(ctx.order); // ensure isoparametric!
   int dim = m.Dimension();

   // Partition the mesh
   ParMesh pmesh(MPI_COMM_WORLD, m);

   Array<int> ref;
   Array<double> l2_error;
   Array<double> h1_error;
   // Refine the mesh
   for (int i = 0; i < ctx.rs; i++)
   {

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
      amg.SetPrintLevel(0);

      // Initialize solver
      HyprePCG pcg(*K_mat);
      pcg.SetTol(1e-8);
      pcg.SetMaxIter(100000);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(amg);
      
      // Solve Ku=f
      pcg.Mult(F_vec, W_gf.GetTrueVector());
      W_gf.SetFromTrueVector();

      FunctionCoefficient exact_coeff(w_exact);
      VectorFunctionCoefficient exact_grad_coeff(2, grad_w_exact);

      // // Write the output
      // ParaViewDataCollection pvdc("KirchoffLove", &pmesh);
      // pvdc.SetHighOrderOutput(true);
      // pvdc.RegisterField("Deformation", &W_gf);

      // ParGridFunction exact_gf(&fespace);
      // exact_gf.ProjectCoefficient(exact_coeff);
      // pvdc.RegisterField("Exact", &exact_gf);
      // pvdc.Save();

      double err_L2 = W_gf.ComputeL2Error(exact_coeff);
      double err_H1 = W_gf.ComputeH1Error(&exact_coeff, &exact_grad_coeff);
      if (rank == 0)
      {
         cout << "-------------------------------------------------------" << endl;
         cout << "RS: " << i << endl;
         int num_iterations; pcg.GetNumIterations(num_iterations);
         cout << "Num Iterations: " << num_iterations << endl; 
         cout << "L2 Error: " << err_L2 << endl;
         cout << "H1 Error: " << err_H1 << endl;
      }
      
      ref.Append(i);
      l2_error.Append(err_L2);
      h1_error.Append(err_H1);
      pmesh.UniformRefinement();
   }
   if (rank == 0)
   {
      cout << "=======================================================" << endl;
      cout << "SUMMARY:" << endl;
      cout << "Eta: " << ctx.eta << endl;
      cout << "Order: " << ctx.order << endl;
      cout << "Refinements: "; ref.Print(cout, ctx.rs);
      cout << "L2 Errors: "; l2_error.Print(cout, ctx.rs);
      cout << "H1 Errors: "; h1_error.Print(cout, ctx.rs);
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

void C0InteriorPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int dim = el1.GetDim();
   MFEM_ASSERT(dim == 2, "Dimension must be 2.");

   int ndof[2] = {el1.GetDof(), 0};
   int num_elems = 1;
   if (Trans.Elem2No >= 0)
   {
      ndof[1] = el2.GetDof();
      num_elems++;
   }

   for (int i = 0; i < num_elems; i++)
   {
      normal[i].SetSize(dim);
      dshape[i].SetSize(ndof[i], dim);
      hessian[i].SetSize(ndof[i], dim * (dim + 1) / 2);
      nv[i].SetSize(dim * (dim + 1) / 2);
      dnshape[i].SetSize(ndof[i]);
      nd2nshape[i].SetSize(ndof[i]);
   }

   for (int i = 0; i < num_elems; i++)
   {
      for (int j = 0; j < num_elems; j++)
      {
         blockJ[i][j].SetSize(ndof[i], ndof[j]);
         blockC[i][j].SetSize(ndof[i], ndof[j]);
      }
   }

   elmatJ_p.SetSize(ndof[0] + ndof[1]);
   elmatC_p.SetSize(ndof[0] + ndof[1]);
   elmat.SetSize(ndof[0] + ndof[1]);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * max(el1.GetOrder(), ndof[1] ? el2.GetOrder() : 0);
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // Compute edge length
   double h_e  = 0.0;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetAllIntPoints(&ip);
      h_e += ip.weight * Trans.Weight();
   }

   const FiniteElement *els[2] = {&el1, &el2};
   ElementTransformation *el_trans[2] = {Trans.Elem1, Trans.Elem2};

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      elmatJ_p = 0.0;
      elmatC_p = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Compute normal gradients + Hessians
      for (int i = 0; i < num_elems; i++)
      {
         if (i == 0)
         {
            CalcOrtho(Trans.Jacobian(), normal[i]);
            normal[i] /= normal[i].Norml2();
         }
         else
         {
            normal[i] = normal[0];
            normal[i] *= -1;
         }
         els[i]->CalcPhysDShape(*el_trans[i], dshape[i]);
         els[i]->CalcPhysHessian(*el_trans[i], hessian[i]);
         dshape[i].Mult(normal[i], dnshape[i]);
         nv[i][0] = normal[i][0]*normal[i][0];
         nv[i][1] = 2*normal[i][0]*normal[i][1];
         nv[i][2] = normal[i][1]*normal[i][1];
         hessian[i].Mult(nv[i], nd2nshape[i]);
      }

      // Compute blocks
      for (int i = 0; i < num_elems; i++)
      {
         for (int j = 0; j < num_elems; j++)
         {  
            blockJ[i][j] = 0.0;
            blockC[i][j] = 0.0;
            AddMult_a_VWt(1.0, dnshape[i], nd2nshape[j], blockJ[i][j]);
            elmatJ_p.SetSubMatrix(i*ndof[0], j*ndof[0], blockJ[i][j]);
            
            AddMult_a_VWt(eta/h_e, dnshape[i], dnshape[j], blockC[i][j]);
            elmatC_p.SetSubMatrix(i*ndof[0], j*ndof[0], blockC[i][j]);
         }
      }

      // Symmetrize the jump term
      elmatJ_p.Symmetrize();
      if (!ndof[1])
      {
         elmatJ_p *= 2;
      }

      // Then just add penalty
      elmatJ_p *= -1;
      elmatJ_p += elmatC_p;
      elmatJ_p *= ip.weight * Trans.Weight();
      elmat += elmatJ_p;
   }
}