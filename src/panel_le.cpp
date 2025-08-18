#include <mfem.hpp>
#include "common.hpp"

using namespace mfem;
using namespace std;

// Define variables for miniapp settings
struct LE_Context
{
   double Lx = 1.0;
   double Ly = 1.0;
   double t   = 0.1;

   int Nx  = 10;
   int Ny  = 10;
   int Nz  = 10;

   int rs    = 0;
   int order = 2;

   // Material properties for 17-4PH stainless steel
   double E  = 196.5e9;
   double nu = 0.27;

   double delta_p_uniform = 1e3;


} ctx;

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
   args.AddOption(&ctx.Nz, "-Nz", "--numelems-z", "Number of elements through panel thickness");

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
   Mesh m = Mesh::MakeCartesian3D(ctx.Nx, ctx.Ny, ctx.Nz, Element::Type::HEXAHEDRON, ctx.Lx, ctx.Ly, ctx.t);
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
   ParFiniteElementSpace fespace(&pmesh, &fe_coll, dim);

   // Get the degrees-of-freedom (DOFs) associated with the sides of the panel
   // (these will be fixed/clamped)
   Array<int> sides_bdr_marker(pmesh.bdr_attributes.Size());
   sides_bdr_marker = 1; // Mark all sides
   sides_bdr_marker[0] = 0; // Unmark z = 0
   sides_bdr_marker[5] = 0; // Unmark z = t

   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(sides_bdr_marker, ess_tdof_list);
   // ess_tdof_list now holds indices of the DOFs associated with the panel sides

   // Create a coefficient representing the uniform pressure BC
   VectorArrayCoefficient p_load(dim);
   p_load.Set(2, new ConstantCoefficient(-ctx.delta_p_uniform)); // set z-th component

   // Convert E and nu to Lame's first and second parameters, as coefficients
   ConstantCoefficient lambda(GetLambda(ctx.E, ctx.nu));
   ConstantCoefficient mu(GetMu(ctx.E, ctx.nu));

   // Initialize the bilinear form representing the LHS (stiffness matrix K in Ku=f)
   ParBilinearForm k(&fespace);
   k.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));

   // Initialize the linear form representing the LHS (forcing term f in Ku=f)
   ParLinearForm f(&fespace);
   Array<int> top_bdr_marker(pmesh.bdr_attributes.Size());
   top_bdr_marker = 0;
   top_bdr_marker[5] = 1;
   f.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(p_load), top_bdr_marker);

   // Initialize the solution
   ParGridFunction U_gf(&fespace);
   U_gf = 0.0;
   U_gf.SetTrueVector();

   // Assemble the stiffness matrix
   k.Assemble();
   k.Finalize();
   std::unique_ptr<HypreParMatrix> K_mat(k.ParallelAssemble());

   // Assemble the RHSs
   Vector F_vec(fespace.GetTrueVSize());
   f.Assemble();
   f.ParallelAssemble(F_vec);

   // Apply boundary conditions to the matrix system
   std::unique_ptr<HypreParMatrix> K_e(K_mat->EliminateRowsCols(ess_tdof_list));
   K_mat->EliminateBC(*K_e, ess_tdof_list, U_gf.GetTrueVector(), F_vec);

   // Initialize preconditioner
   HypreBoomerAMG amg(*K_mat);
   amg.SetElasticityOptions(&fespace);

   // Initialize solver
   HyprePCG pcg(*K_mat);
   pcg.SetTol(1e-8);
   pcg.SetMaxIter(1000);
   pcg.SetPrintLevel(2);
   pcg.SetPreconditioner(amg);

   // Solve Ku=f
   pcg.Mult(F_vec, U_gf.GetTrueVector());
   U_gf.SetFromTrueVector();

   // Write the output
   ParaViewDataCollection pvdc("LinearElasticity", &pmesh);
   pvdc.RegisterField("Displacement", &U_gf);
   pvdc.Save();

   return 0;
}