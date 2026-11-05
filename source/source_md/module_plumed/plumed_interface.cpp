#include "plumed_interface.h"

#include "plumed_atoms.h"

#include <cmath>

#include "source_base/tool_quit.h"

#ifdef USE_PLUMED
// C API of the PLUMED wrapper library, matching <plumed/wrapper/Plumed.h> in a
// PLUMED installation.  The handle is an opaque pointer-sized structure; the
// wrapper library takes care of loading the PLUMED kernel.
extern "C"
{
    plumed_handle plumed_create();
    void plumed_cmd(plumed_handle plumed, const char* key, const void* val);
    void plumed_finalize(plumed_handle plumed);
}
#endif

namespace ModulePlumed
{

namespace
{
// Conversion factors from ABACUS atomic units to the PLUMED default units
// (length: nm, energy: kJ/mol, time: ps, mass: amu).  They are declared to
// PLUMED once in init() with setMD*Units; afterwards all the arrays handed
// over to PLUMED are in ABACUS native units.
constexpr double BOHR_TO_NM = 0.052917721067;             // 1 Bohr = 0.052917721067 nm
constexpr double HARTREE_TO_KJMOL = 2625.499638;          // 1 Hartree = 2625.499638 kJ/mol
constexpr double AU_TIME_TO_PS = 2.418884326509e-5;       // atomic unit of time = 2.418884326509e-5 ps
constexpr double ELECTRON_MASS_TO_AMU = 5.48579909065e-4; // 1 electron mass = 5.48579909065e-4 amu
} // namespace

PlumedInterface::~PlumedInterface()
{
    this->finalize();
}

bool PlumedInterface::init(const std::string& plumed_file,
                           const int nat,
                           const double dt_au,
                           const void* mpi_comm)
{
#ifdef USE_PLUMED
    this->active_run_ = true;
    this->nat_ = nat;

    this->plumed_ = plumed_create();
    this->has_plumed_ = true;
    if (this->plumed_.p == nullptr)
    {
        ModuleBase::WARNING_QUIT("ModulePlumed::PlumedInterface::init",
                                 "could not create a PLUMED object, check the PLUMED installation");
    }

    int real_precision = 8; // double precision reals
    int natoms = nat;
    double length_units = BOHR_TO_NM;
    double energy_units = HARTREE_TO_KJMOL;
    double time_units = AU_TIME_TO_PS;
    double timestep = dt_au;
    const char* engine = "abacus";

    // commands that configure PLUMED must be issued before "init"
    plumed_cmd(this->plumed_, "setRealPrecision", &real_precision);
    // atoms distributed over several ranks: PLUMED runs in its
    // domain-decomposition mode (setAtomsNlocal / setAtomsGatindex every
    // step).  Not handed over for a single rank, so that a PLUMED built
    // without MPI still works there.
    if (mpi_comm != nullptr)
    {
        plumed_cmd(this->plumed_, "setMPIComm", mpi_comm);
    }
    plumed_cmd(this->plumed_, "setMDLengthUnits", &length_units);
    plumed_cmd(this->plumed_, "setMDEnergyUnits", &energy_units);
    plumed_cmd(this->plumed_, "setMDTimeUnits", &time_units);
    plumed_cmd(this->plumed_, "setPlumedDat", plumed_file.c_str());
    plumed_cmd(this->plumed_, "setNatoms", &natoms);
    plumed_cmd(this->plumed_, "setMDEngine", engine);
    plumed_cmd(this->plumed_, "setTimestep", &timestep);
    plumed_cmd(this->plumed_, "init", nullptr);

    return true;
#else
    (void)plumed_file;
    (void)nat;
    (void)dt_au;
    (void)mpi_comm;

    ModuleBase::WARNING_QUIT("ModulePlumed::PlumedInterface::init",
                             "plumed was requested in INPUT but ABACUS was not compiled with PLUMED support; "
                             "reconfigure with -DENABLE_PLUMED=ON");
    return false;
#endif
}

void PlumedInterface::prepare_virial_(const double* cell, const double* virial)
{
    // The Box (cell) data receives forces whenever a periodic collective
    // variable is biased, so PLUMED always needs a valid force buffer here;
    // when the MD run does not provide a virial, a zero buffer is registered
    // instead (as in the reference implementation distributed with GROMACS).
    // The virial is handed over in energy units with the convention
    // virial_ij = -stress_ij * volume (the same quantity used by the official
    // Quantum ESPRESSO interface); ABACUS keeps the potential part of the
    // stress (Hartree/Bohr^3) in its MD virial matrix, so the volume is
    // multiplied back here.
    for (int i = 0; i < 9; ++i)
    {
        this->virial_buf_[i] = 0.0;
    }
    if (virial == nullptr || cell == nullptr)
    {
        return;
    }
    const double volume = std::fabs(cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
                                    - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
                                    + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]));
    for (int i = 0; i < 9; ++i)
    {
        this->virial_buf_[i] = -virial[i] * volume;
    }
}

void PlumedInterface::compute_forces_(const int istep,
                                      OwnedAtoms& owned,
                                      const double* cell,
                                      const double potential,
                                      const double* virial)
{
#ifdef USE_PLUMED
    int step = istep;
    int nlocal = static_cast<int>(owned.gatindex.size());
    double energy = potential;

    // ABACUS stores atomic masses in atomic units (electron masses), while
    // PLUMED expects them in amu; the owned atoms change between steps, so
    // the masses are converted every step
    this->masses_amu_.resize(nlocal);
    for (int i = 0; i < nlocal; ++i)
    {
        this->masses_amu_[i] = owned.masses[i] * ELECTRON_MASS_TO_AMU;
    }

    // a rank may own no atoms; PLUMED is handed valid (unused) buffers then
    int no_index = 0;
    double no_vector[3] = {0.0, 0.0, 0.0};
    double no_mass = 0.0;
    const int* gatindex = &no_index;
    double* positions = no_vector;
    double* forces = no_vector;
    double* masses = &no_mass;
    if (nlocal > 0)
    {
        // Vector3<double> is a plain {double x, y, z} structure, so the
        // arrays can be handed to PLUMED as contiguous [nlocal][3] buffers
        gatindex = owned.gatindex.data();
        positions = reinterpret_cast<double*>(owned.positions.data());
        forces = reinterpret_cast<double*>(owned.forces.data());
        masses = this->masses_amu_.data();
    }

    plumed_cmd(this->plumed_, "setStep", &step);
    // the atoms owned by this rank and their global 0-based indices; with a
    // single rank owning all atoms the indices are the identity
    plumed_cmd(this->plumed_, "setAtomsNlocal", &nlocal);
    plumed_cmd(this->plumed_, "setAtomsGatindex", gatindex);
    plumed_cmd(this->plumed_, "setPositions", positions);
    plumed_cmd(this->plumed_, "setMasses", masses);
    plumed_cmd(this->plumed_, "setBox", cell);
    plumed_cmd(this->plumed_, "setEnergy", &energy);
    // prepare the calculation graph; the arrays that receive the biasing
    // forces are registered afterwards, as done in the reference
    // implementations distributed with PLUMED
    plumed_cmd(this->plumed_, "prepareCalc", nullptr);
    plumed_cmd(this->plumed_, "setForces", forces);
    this->prepare_virial_(cell, virial);
    plumed_cmd(this->plumed_, "setVirial", this->virial_buf_);
    // flag this step as a checkpointing step: PLUMED then flushes all its
    // output files (COLVAR, ...) in update(), so that the collective
    // variables can be followed while the MD is running.  Without it,
    // PLUMED only flushes every 10000 steps.
    int check_point = 1;
    plumed_cmd(this->plumed_, "doCheckPoint", &check_point);
    plumed_cmd(this->plumed_, "performCalc", nullptr);
#else
    (void)istep;
    (void)owned;
    (void)cell;
    (void)potential;
    (void)virial;
#endif
}

void PlumedInterface::move(const int istep,
                           OwnedAtoms& owned,
                           const double* cell,
                           const double potential,
                           const double* virial)
{
#ifdef USE_PLUMED
    if (!this->active_run_)
    {
        return;
    }
    this->compute_forces_(istep, owned, cell, potential, virial);
#else
    (void)istep;
    (void)owned;
    (void)cell;
    (void)potential;
    (void)virial;
#endif
}

void PlumedInterface::finalize()
{
#ifdef USE_PLUMED
    if (this->has_plumed_)
    {
        plumed_finalize(this->plumed_);
        this->has_plumed_ = false;
    }
#endif
    this->active_run_ = false;
}

} // namespace ModulePlumed
