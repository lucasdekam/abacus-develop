#include "plumed_atoms.h"

#include "source_cell/module_neighlist/local_atom.h"

namespace ModulePlumed
{

void collect_owned_atoms(const std::vector<LocalAtom>& atoms,
                         const double lat0,
                         const std::vector<std::int64_t>& type_atom_counts,
                         OwnedAtoms& owned)
{
    // first global index of each species
    std::vector<std::int64_t> type_offset(type_atom_counts.size() + 1, 0);
    for (std::size_t it = 0; it < type_atom_counts.size(); ++it)
    {
        type_offset[it + 1] = type_offset[it] + type_atom_counts[it];
    }

    const std::size_t nlocal = atoms.size();
    owned.gatindex.resize(nlocal);
    owned.positions.resize(nlocal);
    owned.masses.resize(nlocal);
    owned.forces.resize(nlocal);
    for (std::size_t i = 0; i < nlocal; ++i)
    {
        const LocalAtom& atom = atoms[i];
        owned.gatindex[i] = static_cast<int>(type_offset[atom.type] + atom.type_index);
        owned.positions[i] = atom.cart * lat0;
        owned.masses[i] = atom.mass;
        owned.forces[i] = atom.force;
    }
}

void apply_owned_forces(const OwnedAtoms& owned, std::vector<LocalAtom>& atoms)
{
    for (std::size_t i = 0; i < atoms.size(); ++i)
    {
        atoms[i].force = owned.forces[i];
    }
}

} // namespace ModulePlumed
