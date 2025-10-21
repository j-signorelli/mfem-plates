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


void ExactPhysHessian(DenseMatrix &hess)
{
   hess = 0.0;
   
   hess(0,2) = 4;

   hess(1,0) = 4;

   hess(2,0) = 4;
   hess(2,1) = -4;
   hess(2,2) = 4;

   hess(3,1) = -4;

   hess(4,0) = -8;
   hess(4,1) = 4;
   
   hess(5,1) = 4;
   hess(5,2) = -8;
}

void ExactPhysDShape(const Vector &x_vec, DenseMatrix &dshape)
{
   double x = x_vec[0];
   double y = x_vec[1];

   dshape(0,0) = 0;
   dshape(0,1) = 4*y-3;

   dshape(1,0) = 4*x-1;
   dshape(1,1) = 0;

   dshape(2,0) = 4*x-4*y+1;
   dshape(2,1) = -4*x+4*y-1;

   dshape(3,0) = -4*y+4;
   dshape(3,1) = -4*x;

   dshape(4,0) = -8*x+4*y;
   dshape(4,1) = 4*x;

   dshape(5,0) = 4*y-4;
   dshape(5,1) = 4*x-8*y+4;
}

int main(int argc, char *argv[])
{
   int order = 2;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Element order.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Generate mesh
   Mesh m = Mesh::MakeCartesian2D(1,1,Element::Type::TRIANGLE, true);
   m.SetCurvature(order); // ensure isoparametric!

   // Initialize the FE collection and FiniteElementSpace
   H1_FECollection fe_coll(order, 2);
   FiniteElementSpace fespace(&m, &fe_coll, 1);

   GridFunction W_gf(&fespace);
   W_gf = 0.0;

   if(visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << m << W_gf << flush;
      sol_sock << "keys " << "cRnnnnm" << flush; // using dofs_numbering GLVis branch
   }

   string LINE = "\n-------------------------------------------------------\n";
   cout << "GRADIENT + HESSIAN VERIFICATIONS" << LINE;
   {
      FaceElementTransformations &trans = *m.GetInteriorFaceTransformations(0);
      const FiniteElement &fe1 = *fespace.GetFE(trans.Elem1No);
      ElementTransformation &trans_1 = *trans.Elem1;

      // Define integration rule to check all nodes
      // Match reference tri
      IntegrationRule ir(6);
      
      // DOF 0:
      ir.IntPoint(0).x = 0.0;
      ir.IntPoint(0).y = 0.0;

      // DOF 1:
      ir.IntPoint(1).x = 1.0;
      ir.IntPoint(1).y = 0.0;

      // DOF 2:
      ir.IntPoint(2).x = 0.0;
      ir.IntPoint(2).y = 1.0;

      // DOF 3:
      ir.IntPoint(3).x = 0.5;
      ir.IntPoint(3).y = 0.0;

      // DOF 4:
      ir.IntPoint(4).x = 0.5;
      ir.IntPoint(4).y = 0.5;

      // DOF 5:
      ir.IntPoint(5).x = 0.0;
      ir.IntPoint(5).y = 0.5;

      cout << endl << "Checking physical gradient evaluation over Elem " << trans.Elem1No << ":" << endl;
      DenseMatrix physDShape_num(fe1.GetDof(), 2);
      DenseMatrix physDShape_ex(fe1.GetDof(), 2);
      DenseMatrix physDShape_diff(fe1.GetDof(), 2);
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         trans_1.SetIntPoint(&ip);
         fe1.CalcPhysDShape(trans_1, physDShape_num);
         
         Vector coords;
         trans_1.Transform(ip, coords);
         ExactPhysDShape(coords, physDShape_ex);
         
         physDShape_diff = physDShape_num;
         physDShape_diff -= physDShape_ex;

         cout << "\t\\hat{IP}(" << ip.x << "," << ip.y << ") Max Error: " << physDShape_diff.MaxMaxNorm() << endl;
      }

      cout << endl << "Checking physical Hessian evaluation over Elem " << trans.Elem1No << ":" << endl;
      DenseMatrix physHess_num(fe1.GetDof(), 3);
      DenseMatrix physHess_ex(fe1.GetDof(), 3);
      DenseMatrix physHess_diff(fe1.GetDof(), 3);
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         trans_1.SetIntPoint(&ip);
         fe1.CalcPhysHessian(trans_1, physHess_num);

         ExactPhysHessian(physHess_ex);

         physHess_diff = physHess_num;
         physHess_diff -= physHess_ex;

         cout << "\t\\hat{IP}(" << ip.x << "," << ip.y << ") Max Error:" << physHess_diff.MaxMaxNorm() << endl;
      }
   }

   cout << endl << "TERM 1: BIHARMONIC" << LINE;
   {
      // Set-up a 1 point quadrature rule for tris
      IntegrationRule one_point(1);
      one_point.IntPoint(0).x = 1.0/3.0;
      one_point.IntPoint(0).y = 1.0/3.0;
      one_point.IntPoint(0).weight = 0.5; // Reference area is 0.5


      ConstantCoefficient one(1.0);
      BilinearForm a_1(&fespace);
      a_1.AddDomainIntegrator(new BiharmonicIntegrator(one));
      a_1.GetDBFI()->operator[](0)->SetIntRule(&one_point);
      a_1.Assemble(0);
      a_1.Finalize(0);

      cout << endl << "Biharmonic Matrix:" << endl;
      DenseMatrix *a_1_mat = a_1.SpMat().ToDenseMatrix();
      a_1_mat->PrintMatlab(mfem::out);
      delete a_1_mat;
   }

   cout << endl << "TERM 2: JUMP MATRIX" << LINE;
   {
      // Set-up a 1 point quadrature rule for edges
      IntegrationRule one_point_e(1);
      one_point_e.IntPoint(0).x = 0.5;
      one_point_e.IntPoint(0).weight = 1; // Reference edge length is likely just 1
      

      // Set-up Simpson quadrature rule for edges
      IntegrationRule simpson(3);
      simpson.IntPoint(0).x = 0.0;
      simpson.IntPoint(0).weight = 1.0/6.0;
      simpson.IntPoint(1).x = 0.5;
      simpson.IntPoint(1).weight = 2.0/3.0;
      simpson.IntPoint(2).x = 1.0;
      simpson.IntPoint(2).weight = 1.0/6.0;

      BilinearForm a_2(&fespace);
      a_2.AddInteriorFaceIntegrator(new C0InteriorPenaltyIntegrator(1.0));
      //a_2.AddBdrFaceIntegrator(new C0InteriorPenaltyIntegrator(1.0));
      a_2.GetFBFI()->operator[](0)->SetIntRule(&one_point_e);
      //a_2.GetBFBFI()->operator[](0)->SetIntRule(&one_point_e);
      a_2.Assemble(0);
      a_2.Finalize(0);

      cout << endl << "Jump Matrix:" << endl;
      DenseMatrix *a_2_mat = a_2.SpMat().ToDenseMatrix();
      a_2_mat->PrintMatlab(mfem::out);
      delete a_2_mat;
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

   cout << "dnshape_a: "; dnshape_b.Print();

   // dshape_b = (dof, dim)
   dshape_b.Mult(n_b, dnshape_b);

   // hessian_b = (dof, [3 in 2D])
   nv[0] = n_b[0]*n_b[0];
   nv[1] = 2*n_b[0]*n_b[1];
   nv[2] = n_b[1]*n_b[1];

   hessian_b.Mult(nv, nd2nshape_b);

   cout << "nd2nshape_b: "; nd2nshape_b.Print();

   // Consistency term:
   AddMult_a_VWt(0.5, dnshape_a, nd2nshape_b, elmat_ab);

   // Penalty term (symmetric):
   //AddMult_a_VWt(1.0, dnshape_a, dnshape_b, elmat_ab);
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
      //elmat_p.Symmetrize();


      elmat_p *= ip.weight * Trans.Weight();

      elmat += elmat_p;
   }
}