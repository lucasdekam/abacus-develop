#ifndef PLUMED_ATOMS_H
#define PLUMED_ATOMS_H

#include <cstdint>
#include <vector>

#include "source_base/vector3.h"

struct LocalAtom;

namespace ModulePlumed
{

/**
 * @brief the atoms owned by one MPI rank, in the layout PLUMED takes
 *
 * ABACUS distributes the atoms of an MD run over the ranks
 * (MDCell::owned_atoms), so PLUMED runs in its domain-decomposition mode:
 * every step each rank hands over only the atoms it owns, together with
 * their global indices.  With a single rank all atoms are owned by it.
 */
struct OwnedAtoms
{
    std::vector<int> gatindex;                          ///< global 0-based index (STRU order) of each owned atom
    std::vector<ModuleBase::Vector3<double>> positions; ///< Cartesian positions in Bohr
    std::vector<double> masses;                         ///< masses in ABACUS atomic units (electron masses)
    std::vector<ModuleBase::Vector3<double>> forces;    ///< forces in Hartree/Bohr
};

/**
 * @brief collect the atoms owned by this rank for PLUMED
 *
 * The global index of an atom is its position in the STRU file, i.e. the
 * number of atoms of the preceding species plus its index within its own
 * species; PLUMED numbers atoms the same way (starting from 1 in its input).
 *
 * @param atoms            [in] atoms owned by this rank (MDCell::owned_atoms)
 * @param lat0             [in] lattice constant in Bohr; LocalAtom::cart is in
 *                         units of it
 * @param type_atom_counts [in] number of atoms of each species, over all ranks
 * @param owned            [out] the owned atoms in the layout PLUMED takes
 */
void collect_owned_atoms(const std::vector<LocalAtom>& atoms,
                         const double lat0,
                         const std::vector<std::int64_t>& type_atom_counts,
                         OwnedAtoms& owned);

/**
 * @brief write the (PLUMED-biased) forces back into the owned atoms
 *
 * @param owned [in] the owned atoms collected by collect_owned_atoms(), in
 *              the same order as atoms
 * @param atoms [in,out] atoms owned by this rank (MDCell::owned_atoms)
 */
void apply_owned_forces(const OwnedAtoms& owned, std::vector<LocalAtom>& atoms);

} // namespace ModulePlumed

#endif // PLUMED_ATOMS_H
