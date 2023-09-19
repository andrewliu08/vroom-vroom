use std::f64::consts::PI;

use nalgebra as na;
use rand::RngCore;

use lib_reinforcement_learning::genetic_algorithm as ga;

use crate::animal::{Animal, AnimalIndividual};
use crate::world::World;

const GENERATION_STEPS: u32 = 1000;
const MIN_SPEED: f64 = 0.001;
const MAX_SPEED: f64 = 0.005;
const MAX_ACCEL: f64 = 0.2;
const MAX_ANGULAR_ACCEL: f64 = PI / 2.0;

pub struct Simulation {
    world: World,
    evolver: ga::GeneticAlgorithm<
        ga::FitnessProportionateSelection,
        ga::UniformCrossover,
        ga::GaussianMutation,
    >,
    generation: u32,
    generation_steps: u32,
}

impl Simulation {
    pub fn random(rng: &mut dyn RngCore, num_animals: u8, num_food: u8) -> Self {
        let evolver = ga::GeneticAlgorithm::new(
            ga::FitnessProportionateSelection::new(),
            ga::UniformCrossover::new(),
            ga::GaussianMutation::new(0.01, 0.2),
        );

        Self {
            world: World::random(rng, num_animals, num_food),
            evolver,
            generation: 0,
            generation_steps: 0,
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn generation_steps(&self) -> u32 {
        self.generation_steps
    }

    pub fn process_brains(&mut self) {
        for animal in &mut self.world.animals {
            let vision =
                animal
                    .eye
                    .process_vision(animal.position, animal.rotation, &self.world.food);
            let output = animal.brain.forward(vision);

            let speed_accel = output[0].clamp(-MAX_ACCEL, MAX_ACCEL);
            let angular_accel = output[1].clamp(-MAX_ANGULAR_ACCEL, MAX_ANGULAR_ACCEL);
            animal.speed = (animal.speed + speed_accel).clamp(MIN_SPEED, MAX_SPEED);
            animal.rotation = na::Rotation2::new(animal.rotation.angle() + angular_accel);
        }
    }

    pub fn move_animals(&mut self) {
        for animal in &mut self.world.animals {
            // Unit vector for default direction is (1.0, 0.0)
            let displacement = animal.rotation * na::Vector2::x() * animal.speed;
            animal.position += displacement;
            animal.position.x = na::wrap(animal.position.x, 0.0, 1.0);
            animal.position.y = na::wrap(animal.position.y, 0.0, 1.0);
        }
    }

    pub fn eat_food(&mut self, rng: &mut dyn RngCore) {
        const ANIMAL_SIZE: f64 = 0.015;
        const FOOD_SIZE: f64 = 0.005;

        for animal in &mut self.world.animals {
            for food in &mut self.world.food {
                let dist = na::distance(&animal.position, &food.position);
                if dist < ANIMAL_SIZE + FOOD_SIZE {
                    animal.consumed += 1;
                    food.randomize_position(rng);
                }
            }
        }
    }

    pub fn step(&mut self, rng: &mut dyn RngCore) {
        self.generation_steps += 1;
        if self.generation_steps >= GENERATION_STEPS {
            self.evolve(rng);
        }

        self.eat_food(rng);
        self.process_brains();
        self.move_animals();
    }

    pub fn evolve(&mut self, rng: &mut dyn RngCore) {
        self.generation += 1;
        self.generation_steps = 0;

        let curr_population: Vec<AnimalIndividual> = self
            .world
            .animals
            .iter()
            .map(|animal| AnimalIndividual::from_animal(animal))
            .collect();

        let new_population: Vec<Animal> = self
            .evolver
            .evolve(rng, &curr_population)
            .into_iter()
            .map(|individual| individual.into_animal(rng))
            .collect();

        self.world.animals = new_population;

        for food in &mut self.world.food {
            food.randomize_position(rng);
        }
    }
}
