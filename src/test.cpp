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
private:

   void SetHessVec2D(const Vector &nor, Vector &nv);

   const double eta;

   // AssembleFaceMatrix Helpers:
   mutable Vector normal[2], dnshape[2], nv[2], nd2nshape[2];
   mutable DenseMatrix dshape[2], hessian[2], blockJ[2][2], blockC[2][2], elmatJ_p, elmatC_p;
public:
   C0InteriorPenaltyIntegrator(double eta_) : eta(eta_) {};

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, DenseMatrix &elmat) override;
};


void Elem0_ExactPhysHessian(DenseMatrix &hess)
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

void Elem1_ExactPhysHessian(DenseMatrix &hess)
{
   Elem0_ExactPhysHessian(hess);
}

void Elem0_ExactPhysDShape(const Vector &x_vec, DenseMatrix &dshape)
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

void Elem0_ExactNormalPhysDShape(const Vector &x_vec, Vector &grad_n)
{
   double x = x_vec[0];
   double y = x_vec[1];

   grad_n[0] = -4*y+3;
   grad_n[1] = 4*x-1;
   grad_n[2] = 8*x-8*y+2;
   grad_n[3] = 4*x-4*y+4;
   grad_n[4] = -12*x+4*y;
   grad_n[5] = -4*x+12*y-8;
   grad_n *= sqrt(2.0)/2.0;
}

void Elem1_ExactPhysDShape(const Vector &x_vec, DenseMatrix &dshape)
{
   double x = x_vec[0];
   double y = x_vec[1];

   dshape(0,0) = 0;
   dshape(0,1) = 4*y-1;

   dshape(1,0) = 4*x-3;
   dshape(1,1) = 0;

   dshape(2,0) = 4*x-4*y-1;
   dshape(2,1) = -4*x+4*y+1;

   dshape(3,0) = -4*y;
   dshape(3,1) = -4*x+4;

   dshape(4,0) = -8*x+4*y+4;
   dshape(4,1) = 4*x-4;

   dshape(5,0) = 4*y;
   dshape(5,1) = 4*x-8*y;
}

void Elem1_ExactNormalPhysDShape(const Vector &x_vec, Vector &grad_n)
{
   double x = x_vec[0];
   double y = x_vec[1];

   grad_n[0] = 4*y-1;
   grad_n[1] = -4*x+3;
   grad_n[2] = -8*x+8*y+2;
   grad_n[3] = -4*x+4*y+4;
   grad_n[4] = 12*x-4*y-8;
   grad_n[5] = 4*x-12*y;
   grad_n *= sqrt(2.0)/2.0;
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
      const FiniteElement &fe2 = *fespace.GetFE(trans.Elem2No);

      ElementTransformation &trans_1 = *trans.Elem1;
      ElementTransformation &trans_2 = *trans.Elem2;

      for (int elem = 0; elem < 2; elem++)
      {
         const FiniteElement &fe = (elem == 0) ? fe1 : fe2;
         ElementTransformation &el_trans = (elem == 0) ? trans_1 : trans_2;
         const IntegrationRule &ir = fe.GetNodes();

         cout << endl << "Checking physical gradient evaluation over Elem " << elem << ":" << endl;
         DenseMatrix physDShape_num(fe.GetDof(), 2);
         DenseMatrix physDShape_ex(fe.GetDof(), 2);
         DenseMatrix physDShape_diff(fe.GetDof(), 2);
         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir.IntPoint(i);
            el_trans.SetIntPoint(&ip);
            fe.CalcPhysDShape(el_trans, physDShape_num);
            
            Vector coords;
            el_trans.Transform(ip, coords);

            if (elem == 0)
            {
               Elem0_ExactPhysDShape(coords, physDShape_ex);
            }
            else
            {
               Elem1_ExactPhysDShape(coords, physDShape_ex);
            }
            

            physDShape_diff = physDShape_num;
            physDShape_diff -= physDShape_ex;

            cout << "IP(" << coords[0] << "," << coords[1] << "):" << endl;
            physDShape_num.PrintMatlab();
            cout << "Max Error: " << physDShape_diff.MaxMaxNorm() << endl << endl;
         }

         cout << endl << "Checking physical Hessian evaluation over Elem " << elem << ":" << endl;
         DenseMatrix physHess_num(fe.GetDof(), 3);
         DenseMatrix physHess_ex(fe.GetDof(), 3);
         DenseMatrix physHess_diff(fe.GetDof(), 3);
         for (int i = 0; i < ir.GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir.IntPoint(i);
            el_trans.SetIntPoint(&ip);
            fe1.CalcPhysHessian(el_trans, physHess_num);

            Vector coords;
            el_trans.Transform(ip, coords);

            if (elem == 0)
            {
               Elem0_ExactPhysHessian(physHess_ex);
            }
            else
            {
               Elem1_ExactPhysHessian(physHess_ex);
            }

            physHess_diff = physHess_num;
            physHess_diff -= physHess_ex;

            cout << "IP(" << coords[0] << "," << coords[1] << ") Max Error: " << physHess_diff.MaxMaxNorm() << endl;
         }

         // Consider midpoint of element interior edge -- compute normal
         IntegrationPoint ip_mid;
         ip_mid.x = 0.5;
         trans.SetAllIntPoints(&ip_mid);
         Vector normal(2);
         CalcOrtho(trans.Jacobian(), normal);
         normal /= normal.Norml2();
         if (elem == 1)
         {
            normal *= -1;
         }

         cout << endl << "Element " << elem << " normal: "; normal.Print(); cout << endl;

         cout << endl << "Checking normal gradient evaluation on Elem " << elem << " edge midpoint:" << endl;
         Vector physNorDShape_num(fe.GetDof());
         Vector physNorDShape_ex(fe.GetDof());

         fe.CalcPhysDShape(el_trans, physDShape_num);
         physDShape_num.Mult(normal, physNorDShape_num);

         Vector coords;
         trans.Transform(ip_mid, coords);
         cout << "Physical Coordinate: (" << coords[0] << "," << coords[1] << ")" << endl;
         if (elem == 0)
         {
            Elem0_ExactNormalPhysDShape(coords, physNorDShape_ex);
         }
         else
         {
            Elem1_ExactNormalPhysDShape(coords, physNorDShape_ex);
         }


         cout << "Numerical: "; physNorDShape_num.Print();
         cout << "Exact: "; physNorDShape_ex.Print();
         cout << "Error: " << physNorDShape_num.DistanceTo(physNorDShape_ex) << endl;


         cout << endl << "Checking normal Hessian evaluation on Elem " << elem << " edge midpoint:" << endl;
         Vector physNorHess_num(fe.GetDof());
         Vector physNorHess_ex(fe.GetDof());

         fe.CalcPhysDShape(el_trans, physDShape_num);
         physDShape_num.Mult(normal, physNorDShape_num);

         trans.Transform(ip_mid, coords);
         cout << "Physical Coordinate: (" << coords[0] << "," << coords[1] << ")" << endl;
         if (elem == 0)
         {
            Elem0_ExactNormalPhysDShape(coords, physNorDShape_ex);
         }
         else
         {
            Elem1_ExactNormalPhysDShape(coords, physNorDShape_ex);
         }


         cout << "Numerical: "; physNorDShape_num.Print();
         cout << "Exact: "; physNorDShape_ex.Print();
         cout << "Error: " << physNorDShape_num.DistanceTo(physNorDShape_ex) << endl;
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
      a_2.AddBdrFaceIntegrator(new C0InteriorPenaltyIntegrator(1.0));
      a_2.GetFBFI()->operator[](0)->SetIntRule(&simpson);
      a_2.GetBFBFI()->operator[](0)->SetIntRule(&simpson);
      a_2.Assemble(0);
      a_2.Finalize(0);

      cout << endl << "Penalty Matrix:" << endl;
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


void C0InteriorPenaltyIntegrator::SetHessVec2D(const Vector &nor, Vector &nv)
{
   nv[0] = nor[0]*nor[0];
   nv[1] = 2*nor[0]*nor[1];
   nv[2] = nor[1]*nor[1];
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
         SetHessVec2D(normal[i], nv[i]);
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
      elmatJ_p = 0.0;
      elmatJ_p += elmatC_p;
      elmatJ_p *= ip.weight * Trans.Weight();
      elmat += elmatJ_p;
   }
}