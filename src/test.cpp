#include <mfem.hpp>

using namespace mfem;
using namespace std;

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


void GradRef_Phi3(const Vector &x_hat, Vector &grad_phi)
{
   grad_phi.SetSize(2);
   grad_phi[0] = -8*x_hat[0]+4-4*x_hat[1];
   grad_phi[1] = -4*x_hat[0];
}

int main(int argc, char *argv[])
{
   int order = 2;
   int rs = 0;
   real_t eval_point = 0.5;
   bool visualization = true;
   real_t eta = 1e3;
   bool quads = true;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Element order.");
   args.AddOption(&rs, "-rs", "--refine-serial", "Number of times to refine mesh in serial.");
   args.AddOption(&eval_point, "-e", "--eval-point", "Evaluation point along face length in reference coordinates.");
   args.AddOption(&quads, "-quad", "--quad-mesh", "-tri",
                  "--tri-mesh",
                  "Set mesh type.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&eta, "-eta", "--penalty-coeff", "Penalty coefficient.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Generate mesh
   Mesh m = Mesh::MakeCartesian2D(1,1,Element::Type::TRIANGLE, true);
   //Mesh m("/home/j-signorelli/software/mfem/git_repo/data/ref-triangle.mesh", 1);
   m.SetCurvature(order); // ensure isoparametric!

   // Refine the mesh
   for (int i = 0; i < rs; i++)
   {
      m.UniformRefinement();
   }

   // Initialize the FE collection and FiniteElementSpace
   H1_FECollection fe_coll(order, 2);
   FiniteElementSpace fespace(&m, &fe_coll, 1);

   // ConstantCoefficient one(1.0);
   // // BilinearForm a_1(&fespace);
   // // a_1.AddDomainIntegrator(new BiharmonicIntegrator(one));
   // // a_1.Assemble(0);
   // // a_1.Finalize(0);

   // // BilinearForm a_2(&fespace);
   // // a_2.AddInteriorFaceIntegrator(new C0InteriorPenaltyIntegrator(order*(order+1)));
   // // a_2.Assemble(0);
   // // a_2.Finalize(0);

   // // Now assemble actual stuff
   // BilinearForm a(&fespace);
   // a.AddDomainIntegrator(new BiharmonicIntegrator(one));
   // a.AddInteriorFaceIntegrator(new C0InteriorPenaltyIntegrator(order*(order+1)));
   // a.Assemble();
   // a.Finalize();
   // DenseMatrix *a_mat = a.SpMat().ToDenseMatrix();
   // a_mat->PrintMatlab(mfem::out);
   // delete a_mat;
   // // cout << "Biharmonic Matrix:" << endl;
   // // DenseMatrix *a_1_mat = a_1.SpMat().ToDenseMatrix();
   // // //a_1_mat->Print(mfem::out, 100);
   // // delete a_1_mat;
   
   // // cout << "C0-IP Matrix:" << endl;
   // // DenseMatrix *a_2_mat = a_2.SpMat().ToDenseMatrix();
   // // //a_2_mat->Print(mfem::out, 100);
   // // delete a_2_mat;

   // LinearForm f(&fespace);
   // f.AddDomainIntegrator(new DomainLFIntegrator(one));
   // f.Assemble();

   GridFunction W_gf(&fespace);
   W_gf = 0.0;
   // W_gf.SetTrueVector();

   // SparseMatrix A;
   // Vector B;
   // Array<int> boundary_dofs;
   // fespace.GetBoundaryTrueDofs(boundary_dofs);
   // a.FormLinearSystem(boundary_dofs, W_gf, f, A, W_gf.GetTrueVector(), B);

   // GSSmoother M(A);
   // PCG(A, M, B, W_gf.GetTrueVector(), 1, 1000000, 1e-8, 0.0);

   // W_gf.SetFromTrueVector();

   if(visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << m << W_gf << flush;
      sol_sock << "keys " << "cRnnnnm" << flush; // using dofs_numbering GLVis branch
   }

   FaceElementTransformations &trans = *m.GetInteriorFaceTransformations(0);
   const FiniteElement &fe1 = *fespace.GetFE(trans.Elem1No);
   const FiniteElement &fe2 = *fespace.GetFE(trans.Elem2No);
   
   cout << "Elem1No: " << trans.Elem1No << endl;
   cout << "Elem2No: " << trans.Elem2No << endl;

   cout << "TERM 1: BIHARMONIC\n-----------------------------------" << endl;

   cout << "Checking reference gradient evaluation of \\phi_3 over Elem " << trans.Elem1No << ":" << endl;
   int biharmonic_integration_order = 2*fe1.GetOrder() + trans.Elem1->OrderW();
   const IntegrationRule *ir = &IntRules.Get(trans.Elem1->GetGeometryType(),
                                             biharmonic_integration_order);
   
   DenseMatrix gradP(fe1.GetDof(), 2);
   Vector grad_phi_3(2), grad_phi_3_exact(2);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      fe1.CalcDShape(ip, gradP);
      gradP.GetRow(3, grad_phi_3);
      GradRef_Phi3(Vector({ip.x, ip.y}), grad_phi_3_exact);
      cout << "\t\\hat{IP}(" << ip.x << "," << ip.y << ") Error: " << grad_phi_3.DistanceTo(grad_phi_3_exact) << endl;
   }


   cout << "Checking reference Hessian evaluation of \\phi_3 over Elem " << trans.Elem1No << ":" << endl;
   DenseMatrix hessian(fe1.GetDof(), 3);
   Vector hessian_phi3(3);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      fe1.CalcHessian(ip, hessian);
      hessian.GetRow(3, hessian_phi3);
      cout << "\t\\hat{IP}(" << ip.x << "," << ip.y << "):"; hessian_phi3.Print(mfem::out, 3);
   }


   /*
   trans.CheckConsistency(1, mfem::out);

   cout << endl << "Face DOFs' Locations: " << endl;
   trans.GetPointMat().Print();

   IntegrationPoint ip;
   ip.x = eval_point;
   trans.SetAllIntPoints(&ip);
   cout << endl << "Eval Point IP: " << ip.x << endl;
   Vector ip_phys;
   std::array<std::pair<ElementTransformation*, int>, 2> elems;
   elems[0].first = trans.Elem1;
   elems[0].second = trans.Elem1No;
   elems[1].first = trans.Elem2;
   elems[1].second = trans.Elem2No;
   for (int i = 0; i < 2; i++)
   {  
      ElementTransformation &elem_T = *elems[i].first;
      int elemNo = elems[i].second;
      const IntegrationPoint &eip = elem_T.GetIntPoint();
      const FiniteElement &fe = *fespace.GetFE(elemNo);

      cout << endl << "Elem" << elemNo << " IP Reference Space: ( " << eip.x << " , " << eip.y << " )" << endl;
      elem_T.Transform(eip, ip_phys);
      cout << "Elem" << elemNo << " IP Physical Space: ( " << ip_phys[0] << " , " << ip_phys[1] << " )" << endl;
      
      
      cout << endl << "CalcShape @ Eval Point: " << endl;
      Vector shape(fe.GetDof());
      fe.CalcShape(eip, shape);
      shape.Print(mfem::out, fe.GetDof());
   }
   */
   /*
   Vector ip1_phys, ip2_phys;
   trans.Elem1->Transform(trans.GetElement1IntPoint(), ip1_phys);
   trans.Elem2->Transform(trans.GetElement2IntPoint(), ip2_phys);

   cout << endl << "IP: " << ip.x << endl;
   cout << endl << "Elem1 IP Reference Space: ( " << trans.GetElement1IntPoint().x << " , " << trans.GetElement1IntPoint().y << " )" << endl;
   cout << "Elem1 IP Physical Space: ( " << ip1_phys[0] << " , " << ip1_phys[1] << " )" << endl;
   cout << endl << "Elem2 IP Reference Space: ( " << trans.GetElement2IntPoint().x << " , " << trans.GetElement2IntPoint().y << " )" << endl;
   cout << "Elem2 IP Physical Space: ( " << ip2_phys[0] << " , " << ip2_phys[1] << " )" << endl;

   cout << endl << "FaceElementTransformation Jacobian: " << endl;
   trans.Jacobian().Print();

   Vector normal_1(2), normal_2(2);
   CalcOrtho(trans.Jacobian(), normal_1);
   normal_1 /= normal_1.Norml2();
   normal_2 = normal_1;
   normal_2 *= -1.0;
   cout << endl << "Normal 1: "; normal_1.Print();
   cout << "Normal 2: "; normal_2.Print();

   const FiniteElement &el1 = *fespace.GetFE(trans.Elem1No);
   const FiniteElement &el2 = *fespace.GetFE(trans.Elem2No);

   int quad_order = 2 * max(el1.GetOrder(), el2.GetOrder());
   cout << endl << "Geometry Type: " << trans.GetGeometryType() << endl;
   cout << "Quadrature Order: " << quad_order << endl;
   const IntegrationRule *ir = &IntRules.Get(trans.GetGeometryType(), quad_order);

   // Compute edge length + print integration points
   double h_e  = 0.0;
   cout << "-------------------------------------------" << endl;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      trans.SetAllIntPoints(&ip);
      cout << endl << endl << "IP" << p << ": " << ip.x << endl;
      cout << "Jacobian @ Eval Point: " << endl;
      trans.Jacobian().Print();
      h_e += ip.weight*trans.Weight();
   }
   cout << "-------------------------------------------" << endl;
   cout << "Edge Length: " << h_e << endl;
   */
   // ElementTransformation &el_trans = *fespace.GetElementTransformation(0);
   // IntegrationPoint ip;
   // ip.Init(0);
   // ip.x = 0.5;
   // ip.y = 0.5;
   // el_trans.SetIntPoint(&ip);
   // cout << endl << "Jacobian @ Eval Point: " << endl;
   // el_trans.Jacobian().Print();

   // cout << endl << "CalcShape @ Eval Point: " << endl;
   // Vector shape(el.GetDof());
   // el.CalcShape(ip, shape);
   // shape.Print(mfem::out, el.GetDof());

   // cout << endl << "CalcDShape @ Eval Point: " << endl;
   // DenseMatrix dshape(el.GetDof(), 2);
   // el.CalcDShape(ip, dshape);
   // dshape.Print(mfem::out, 2);

   // cout << endl << "CalcPhysDShape @ Eval Point: " << endl;
   // DenseMatrix phys_dshape(el.GetDof(), 2);
   // el.CalcPhysDShape(el_trans, phys_dshape);
   // phys_dshape.Print(mfem::out, 2);


   // cout << endl << "CalcHessian @ Eval Point: " << endl;
   // DenseMatrix hess(el.GetDof(), 3);
   // el.CalcHessian(ip, hess);
   // hess.Print(mfem::out, 3);

   // cout << endl << "CalcPhysHessian @ Eval Point: " << endl;
   // DenseMatrix phys_hess(el.GetDof(), 3);
   // el.CalcPhysHessian(el_trans, phys_hess);
   // phys_hess.Print(mfem::out, 3);

   //FaceElementTransformations &face_trans = *fespace.GetBdrElementTransformation()
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
      // cout << "Biharmonic element matrix:" << endl;
      // elmat.Print();
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

   // Consistency term:
   //cout << "Element consistency matrix:" << endl;
   //elmat_ab.Print(mfem::out, 1000);


   // Penalty term (symmetric):
   //cout << "Penalty matrix pre-coefficient application:" << endl;
   //AddMult_a_VWt(1.0, dnshape_a, dnshape_b, elmat_ab);
   //elmat_ab.Print(mfem::out, 100);
   //elmat_ab = 0.0;

   //cout << "Coefficient: " << eta << " / " << h_e << " = " << eta/h_e << endl;
   //cout << "Penalty matrix post-coefficient application:" << endl;
   AddMult_a_VWt(eta/h_e, dnshape_a, dnshape_b, elmat_ab);
   //elmat_ab.Print(mfem::out, 100);
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

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Compute normals, derivatives, and hessians
      CalcOrtho(Trans.Jacobian(), normal_1);
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

      elmat_p *= ip.weight * Trans.Weight();

      elmat += elmat_p;
   }
}