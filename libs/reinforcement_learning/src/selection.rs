pub use self::fitness_proportionate_selection::FitnessProportionateSelection;

use rand::RngCore;

use crate::individual::Individual;

mod fitness_proportionate_selection;

pub trait Selection {
    fn select<'a, I: Individual>(
        &self,
        rng: &mut dyn RngCore,
        population: &'a [I],
        cnt: u32,
    ) -> Vec<&'a I>;
}
