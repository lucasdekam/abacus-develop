#include "source_md/module_plumed/plumed_atoms.h"
#include "source_cell/module_neighlist/local_atom.h"

#include <gtest/gtest.h>

namespace
{
LocalAtom make_atom(const int type, const std::int64_t type_index, const double z)
{
    LocalAtom atom;
    atom.type = type;
    atom.type_index = type_index;
    atom.cart.set(0.5, 0.25, z);
    atom.mass = 1000.0 * (type + 1);
    atom.force.set(0.0, 0.0, 0.1 * z);
    return atom;
}
} // namespace

// a rank owning atoms of two species out of STRU order: the global index is
// the species offset plus the index within the species
TEST(PlumedAtomsTest, CollectOwnedAtomsGlobalIndex)
{
    const std::vector<std::int64_t> type_atom_counts = {2, 3};
    std::vector<LocalAtom> atoms;
    atoms.push_back(make_atom(1, 2, 3.0));
    atoms.push_back(make_atom(0, 1, 1.0));
    atoms.push_back(make_atom(1, 0, 2.0));
    const double lat0 = 1.8897261;

    ModulePlumed::OwnedAtoms owned;
    ModulePlumed::collect_owned_atoms(atoms, lat0, type_atom_counts, owned);

    ASSERT_EQ(owned.gatindex.size(), 3U);
    EXPECT_EQ(owned.gatindex[0], 4);
    EXPECT_EQ(owned.gatindex[1], 1);
    EXPECT_EQ(owned.gatindex[2], 2);
    ASSERT_EQ(owned.positions.size(), 3U);
    EXPECT_DOUBLE_EQ(owned.positions[0].x, 0.5 * lat0);
    EXPECT_DOUBLE_EQ(owned.positions[0].z, 3.0 * lat0);
    EXPECT_DOUBLE_EQ(owned.positions[1].z, 1.0 * lat0);
    ASSERT_EQ(owned.masses.size(), 3U);
    EXPECT_DOUBLE_EQ(owned.masses[0], 2000.0);
    EXPECT_DOUBLE_EQ(owned.masses[1], 1000.0);
    ASSERT_EQ(owned.forces.size(), 3U);
    EXPECT_DOUBLE_EQ(owned.forces[2].z, 0.2);
}

// the biased forces go back to the same owned atoms
TEST(PlumedAtomsTest, ApplyOwnedForces)
{
    const std::vector<std::int64_t> type_atom_counts = {2, 3};
    std::vector<LocalAtom> atoms;
    atoms.push_back(make_atom(1, 2, 3.0));
    atoms.push_back(make_atom(0, 1, 1.0));

    ModulePlumed::OwnedAtoms owned;
    ModulePlumed::collect_owned_atoms(atoms, 1.0, type_atom_counts, owned);
    owned.forces[0].z -= 1.0;
    owned.forces[1].x += 0.5;
    ModulePlumed::apply_owned_forces(owned, atoms);

    EXPECT_DOUBLE_EQ(atoms[0].force.z, 0.3 - 1.0);
    EXPECT_DOUBLE_EQ(atoms[0].force.x, 0.0);
    EXPECT_DOUBLE_EQ(atoms[1].force.x, 0.5);
    EXPECT_DOUBLE_EQ(atoms[1].force.z, 0.1);
}

// a rank that owns no atoms hands over empty arrays
TEST(PlumedAtomsTest, CollectOwnedAtomsEmptyRank)
{
    const std::vector<std::int64_t> type_atom_counts = {2, 3};
    std::vector<LocalAtom> atoms;
    ModulePlumed::OwnedAtoms owned;
    owned.gatindex.assign(4, 7);

    ModulePlumed::collect_owned_atoms(atoms, 1.0, type_atom_counts, owned);

    EXPECT_TRUE(owned.gatindex.empty());
    EXPECT_TRUE(owned.positions.empty());
    EXPECT_TRUE(owned.masses.empty());
    EXPECT_TRUE(owned.forces.empty());
}
