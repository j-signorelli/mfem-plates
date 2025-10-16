#include <mfem.hpp>

using namespace mfem;
using namespace std;



int main(int argc, char *argv[])
{
   int order = 2;
   Mesh m = Mesh::MakeCartesian2D(1,1,Element::Type::TRIANGLE, true);
   m.SetCurvature(order);
   
   H1_FECollection fe_coll(order, 2);
   FiniteElementSpace fespace(&m, &fe_coll, 1);

   ElementTransformation &Tr = *fespace.GetElementTransformation(0);
   const FiniteElement &fe = *fespace.GetFE(0);

   IntegrationPoint test_ip;
   test_ip.x = 0.5;
   test_ip.y = 0.0;
   Tr.SetIntPoint(&test_ip);
   
   cout << "Before calling GetElementTransformation(1):" << endl;
   // DenseMatrix grad_ref(fe.GetDof(), 2);
   // fe.CalcDShape(test_ip, grad_ref);
   // cout << endl << "Grad Ref:" << endl; grad_ref.PrintMatlab();

   // DenseMatrix grad_phys(fe.GetDof(), 2);
   // fe.CalcPhysDShape(Tr, grad_phys);
   // cout << endl << "Grad Phys:" << endl; grad_phys.PrintMatlab();

   cout << endl << "Jacobian:" << endl; Tr.Jacobian().PrintMatlab();

   fespace.GetElementTransformation(1);

   cout << endl << endl << "After calling GetElementTransformation(1):" << endl;
   // fe.CalcDShape(test_ip, grad_ref);
   // cout << endl << "Grad Ref:" << endl; grad_ref.PrintMatlab();

   // fe.CalcPhysDShape(Tr, grad_phys);
   // cout << endl << "Grad Phys:" << endl; grad_phys.PrintMatlab();

   cout << endl << "Jacobian:" << endl; Tr.Jacobian().PrintMatlab();

   return 0;
}