#ifndef PLUMED_INTERFACE_H
#define PLUMED_INTERFACE_H

#include <string>
#include <vector>

/// opaque handle of the PLUMED wrapper library, see <plumed/wrapper/Plumed.h>
/// in a PLUMED installation; it is a plain pointer-sized structure, so it can
/// be stored by value and initialized to a null pointer
struct plumed_handle
{
    void* p;
};

namespace ModulePlumed
{

struct OwnedAtoms;

/**
 * @brief interface between an ABACUS MD run and the PLUMED plugin
 *
 * This class wraps the C API of the PLUMED wrapper library (``libplumed.so``,
 * ``plumed_create`` / ``plumed_cmd`` / ``plumed_finalize``, see
 * <plumed/wrapper/Plumed.h> in a PLUMED installation) so that
 * collective variables, biasing potentials and free-energy methods provided
 * by PLUMED (https://www.plumed.org) can be used in ABACUS MD runs.
 *
 * Units: ABACUS atomic units differ from the PLUMED defaults, so the
 * conversion factors are declared once in init() through the commands
 * ``setMDEnergyUnits`` (Hartree -> kJ/mol), ``setMDLengthUnits``
 * (Bohr -> nm) and ``setMDTimeUnits`` (atomic unit of time -> ps).  From then
 * on all quantities are passed in ABACUS native units and PLUMED performs the
 * conversion internally:
 *   - positions / cell                   : Bohr
 *   - forces                             : Hartree/Bohr (biasing forces are
 *                                          added in place by PLUMED)
 *   - potential energy                   : Hartree
 *   - virial                             : Hartree (=-stress * volume,
 *                                          following the Quantum ESPRESSO
 *                                          interface convention)
 *   - masses                             : electron masses (converted to amu
 *                                          internally)
 *
 * Several MPI ranks: ABACUS distributes the atoms of an MD run over the ranks
 * (MDCell::owned_atoms), so PLUMED then runs in its domain-decomposition
 * mode, as in the LAMMPS and GROMACS interfaces distributed with PLUMED.
 * init() hands over the MD communicator, and every step each rank passes
 * only the atoms it owns with their global indices (OwnedAtoms,
 * ``setAtomsNlocal`` / ``setAtomsGatindex``); PLUMED gathers the atoms its
 * actions need and adds the biasing forces to each rank's own atoms.  The
 * potential energy and the cell are the same on every rank and are passed as
 * such.
 *
 * This interface is a no-op stub unless ABACUS is configured with
 * ``-DENABLE_PLUMED=ON``; requesting PLUMED in INPUT while ABACUS was built
 * without it stops the run with an error message.
 */
class PlumedInterface
{
  public:
    PlumedInterface() = default;
    ~PlumedInterface();

    PlumedInterface(const PlumedInterface&) = delete;
    PlumedInterface& operator=(const PlumedInterface&) = delete;

    /**
     * @brief create a PLUMED object and read the PLUMED input file
     *
     * Must be called once before the MD loop.
     *
     * @param plumed_file [in] path of the PLUMED input file
     * @param nat         [in] number of atoms over all ranks
     * @param dt_au       [in] MD time step in atomic units of time
     * @param mpi_comm    [in] pointer to the MPI_Comm the atoms are
     *                    distributed over, or nullptr when a single rank
     *                    owns all atoms; PLUMED must then be built with MPI
     * @return true when the interface is ready to be used in the MD loop
     */
    bool init(const std::string& plumed_file,
              const int nat,
              const double dt_au,
              const void* mpi_comm);

    /**
     * @brief one MD step: hand over the configuration, let PLUMED compute the
     * collective variables and add the biasing forces to ``force``
     *
     * @param istep     [in] global MD step index
     * @param owned     [in,out] the atoms owned by this rank; PLUMED adds the
     *                  biasing forces to owned.forces in place
     * @param cell      [in] 3x3 cell matrix in Bohr, row-major, the rows being
     *                  the lattice vectors
     * @param potential [in] potential energy in Hartree
     * @param virial    [in] 3x3 virial in Hartree (=-stress * volume), or
     *                  nullptr when the stress is not available
     */
    void move(const int istep,
              OwnedAtoms& owned,
              const double* cell,
              const double potential,
              const double* virial);

    /// @brief finalize PLUMED (flushes the COLVAR and other output files)
    void finalize();

    /// @brief whether PLUMED is active in this run
    bool active() const
    {
        return this->active_run_;
    }

  private:
    /// @brief register the configuration of one step with PLUMED and let it
    /// compute the collective variables and the biasing forces
    void compute_forces_(const int istep,
                         OwnedAtoms& owned,
                         const double* cell,
                         const double potential,
                         const double* virial);

    /// @brief build the 3x3 virial (in energy units) handed over to PLUMED
    void prepare_virial_(const double* cell, const double* virial);

    plumed_handle plumed_{nullptr};  ///< handle returned by plumed_create()
    bool has_plumed_ = false;        ///< whether a PLUMED object is alive
    bool active_run_ = false;        ///< whether init() has been called
    int nat_ = 0;                    ///< number of atoms over all ranks
    std::vector<double> masses_amu_; ///< masses of the owned atoms in amu
    double virial_buf_[9] = {0.0};   ///< virial handed over to PLUMED (energy units)
};

} // namespace ModulePlumed

#endif // PLUMED_INTERFACE_H
