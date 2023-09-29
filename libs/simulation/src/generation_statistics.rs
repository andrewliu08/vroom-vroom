use lib_reinforcement_learning::genetic_algorithm::Individual;

pub struct GenerationStatistics {
    pub max_fitness: f64,
    pub min_fitness: f64,
    pub mean_fitness: f64,
    pub std_fitness: f64,
}

impl GenerationStatistics {
    pub fn from_population<I: Individual>(population: &[I]) -> Self {
        assert!(!population.is_empty());

        let mut max_fitness: f64 = 0.0;
        let mut min_fitness: f64 = 0.0;
        let mut sum_fitness: f64 = 0.0;
        let mut sum_sq_fitness: f64 = 0.0;
        for individual in population {
            max_fitness = max_fitness.max(individual.fitness());
            min_fitness = min_fitness.min(individual.fitness());
            sum_fitness += individual.fitness();
            sum_sq_fitness += individual.fitness().powi(2);
        }

        let mean_fitness = sum_fitness / population.len() as f64;
        let var_fitness = (sum_sq_fitness / population.len() as f64) - mean_fitness.powi(2);

        GenerationStatistics {
            max_fitness,
            min_fitness,
            mean_fitness,
            std_fitness: var_fitness.sqrt(),
        }
    }
}
