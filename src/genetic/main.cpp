#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "GAInput.h"
#include "GAOptimize.h"
#include "fiedler.h"
using namespace std;

#ifndef SERIAL
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

int main(int argc, char* argv[])
{
#ifndef SERIAL
  mpi::environment env(argc, argv);
  mpi::communicator world;
  if(world.rank() == 0) cout << "Parallel GA simulation" << endl;
#endif

  string confFileName;
  string dumpFileName;

  bool simple = false;
  for(int i = 1; i < argc; ++i)
  {
    if(strcmp(argv[i], "-s") == 0) {simple = true;}
    if(strcmp(argv[i], "-config")   == 0) confFileName = argv[++i];
    if(strcmp(argv[i], "-integral") == 0) dumpFileName = argv[++i];
  }
  ifstream confFile(confFileName.c_str());
  ifstream dumpFile(dumpFileName.c_str());

  std::vector<int> fiedlerv = get_fiedler(dumpFileName, dumpFile, simple);

  genetic::Cell final = genetic::gaordering(confFile, dumpFile, fiedlerv, simple);

#ifndef SERIAL
  if(world.rank() == 0)
#endif
  {
    cout << "##################### MINIMUM GENE REP. #####################" << endl;
    cout << "Gene with MinValue = " << final << endl;
    cout << "Effective Distance = " << sqrt(final.Fitness()) << endl;

    cout << "#################### DMRG REORDER FORMAT ####################" << endl;
    int n = genetic::Gene::Length() - 1;
    vector<int> gaorder(final.Gen().Sequence());

    for(int i = 0; i < n; ++i) cout << gaorder[i] + 1 << ",";
    cout << gaorder[n] + 1 << endl;
  }

  return 0;
}
