/*                                                                           
Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012                      
Copyright (c) 2012, Garnet K.-L. Chan                                        
                                                                             
This program is integrated in Molpro with the permission of 
Sandeep Sharma and Garnet K.-L. Chan
*/


#ifndef SWEEPONEPDM_HEADER
#define SWEEPONEPDM_HEADER
using namespace boost;
using namespace std;

namespace SpinAdapted{
  class SweepParams;
  class StackSpinBlock;
namespace SweepOnepdm
{
  void BlockAndDecimate (SweepParams &sweepParams, StackSpinBlock& system, StackSpinBlock& newSystem, const bool &useSlater, const bool& dot_with_sys, int state);
  double do_one(SweepParams &sweepParams, const bool &warmUp, const bool &forward, const bool &restart, const int &restartSize, int state);
};
}

#endif
