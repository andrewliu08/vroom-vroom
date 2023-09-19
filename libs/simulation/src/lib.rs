use std::f64::consts::PI;

use ga::Chromosome;
use nalgebra as na;
use rand::{Rng, RngCore};

use lib_neural_net as nn;
use lib_reinforcement_learning as ga;

const GENERATION_STEPS: u32 = 1500;
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

pub struct World {
    animals: Vec<Animal>,
    food: Vec<Food>,
}

pub struct Animal {
    position: na::Point2<f64>,
    rotation: na::Rotation2<f64>,
    speed: f64,
    consumed: u32,
    eye: Eye,
    brain: nn::MLP,
}

pub struct AnimalIndividual {
    chromosome: ga::Chromosome,
    fitness: f64,
}

pub struct Eye {
    fov_range: f64,
    fov_angle: f64,
    receptors: usize,
}

pub struct Food {
    position: na::Point2<f64>,
}

impl Simulation {
    pub fn random(rng: &mut dyn RngCore, num_animals: u8, num_food: u8) -> Self {
        let evolver = ga::GeneticAlgorithm::new(
            ga::FitnessProportionateSelection::new(),
            ga::UniformCrossover::new(),
            ga::GaussianMutation::new(0.1, 0.2),
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
                    food.position = rng.gen();
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
            food.position = rng.gen();
        }
    }
}

impl World {
    pub fn random(rng: &mut dyn RngCore, num_animals: u8, num_food: u8) -> Self {
        let animals = (0..num_animals).map(|_| Animal::random(rng)).collect();
        let food = (0..num_food).map(|_| Food::random(rng)).collect();
        Self { animals, food }
    }

    pub fn animals(&self) -> &[Animal] {
        &self.animals
    }

    pub fn food(&self) -> &[Food] {
        &self.food
    }
}

impl Animal {
    pub fn new(rng: &mut dyn RngCore, eye: Eye, brain: nn::MLP) -> Self {
        Self {
            position: rng.gen(),
            rotation: rng.gen(),
            speed: 0.001,
            consumed: 0,
            eye,
            brain,
        }
    }

    pub fn random(rng: &mut dyn RngCore) -> Self {
        let eye = Eye::default();
        let brain = nn::MLP::random_weights(rng, eye.receptors, &[2 * eye.receptors, 2], 0.01);
        Self::new(rng, eye, brain)
    }

    pub fn from_chromosome(rng: &mut dyn RngCore, chromosome: ga::Chromosome) -> Self {
        let eye = Eye::default();
        let brain =
            nn::MLP::from_weight_and_biases(eye.receptors, &[2 * eye.receptors, 2], chromosome);
        Self::new(rng, eye, brain)
    }

    pub fn as_chromosome(&self) -> Chromosome {
        ga::Chromosome::new(self.brain.weights_and_biases())
    }

    pub fn position(&self) -> na::Point2<f64> {
        self.position
    }

    pub fn rotation(&self) -> na::Rotation2<f64> {
        self.rotation
    }

    pub fn speed(&self) -> f64 {
        self.speed
    }
}

impl AnimalIndividual {
    fn from_animal(animal: &Animal) -> Self {
        Self {
            chromosome: animal.as_chromosome(),
            fitness: animal.consumed as f64,
        }
    }

    fn into_animal(&self, rng: &mut dyn RngCore) -> Animal {
        Animal::from_chromosome(rng, self.chromosome.clone())
    }
}

impl Eye {
    pub fn new(fov_range: f64, fov_angle: f64, receptors: usize) -> Self {
        Self {
            fov_range,
            fov_angle,
            receptors,
        }
    }

    pub fn default() -> Self {
        Self {
            fov_range: 0.25,
            fov_angle: PI / 2.0,
            receptors: 10,
        }
    }

    pub fn process_vision(
        &self,
        position: na::Point2<f64>,
        rotation: na::Rotation2<f64>,
        food: &[Food],
    ) -> Vec<f64> {
        let angle_per_receptor = self.fov_angle / self.receptors as f64;
        let mut receptors = vec![2.0; self.receptors];

        for f in food {
            let displacement = f.position - position;
            let dist = displacement.norm();
            if dist > self.fov_range {
                continue;
            }

            let angle = na::Rotation2::rotation_between(&na::Vector2::x(), &displacement).angle();
            let angle = na::wrap(angle - rotation.angle(), -PI, PI);
            let angle = angle + self.fov_angle / 2.0;
            if angle < 0.0 || angle > self.fov_angle {
                continue;
            }

            let receptor_idx =
                std::cmp::min((angle / angle_per_receptor) as usize, self.receptors - 1);
            receptors[receptor_idx] = f64::min(receptors[receptor_idx], dist / self.fov_range);
        }

        receptors
    }
}

impl Food {
    fn random(rng: &mut dyn RngCore) -> Self {
        Self {
            position: rng.gen(),
        }
    }

    pub fn position(&self) -> na::Point2<f64> {
        self.position
    }
}

impl ga::Individual for AnimalIndividual {
    fn from_chromosome(chromosome: ga::Chromosome) -> Self {
        Self {
            chromosome,
            fitness: 0.0,
        }
    }

    fn as_chromosome(&self) -> &ga::Chromosome {
        &self.chromosome
    }

    fn fitness(&self) -> f64 {
        self.fitness
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestCase {
        fov_range: f64,
        fov_angle: f64,
        receptors: usize,
        x: f64,
        y: f64,
        rotation: f64,
        food: Vec<Food>,
        expected: &'static str,
    }

    impl TestCase {
        fn run(&self) {
            let eye = Eye::new(self.fov_range, self.fov_angle, self.receptors);

            let actual = eye.process_vision(
                na::Point2::new(self.x, self.y),
                na::Rotation2::new(self.rotation),
                &self.food,
            );
            let actual = actual
                .into_iter()
                .map(|dist| {
                    if dist > 1.0 {
                        " "
                    } else if dist > 0.6 {
                        "."
                    } else if dist > 0.3 {
                        "o"
                    } else {
                        "O"
                    }
                })
                .collect::<Vec<_>>()
                .join("");

            assert_eq!(actual, self.expected);
        }
    }

    mod test_fov_ranges {
        use super::*;

        /*
             /......
           /........
        @>....o.....
           \........
             \......

             /....
           /......
        @>....o...
           \......
             \...

             /.
           /...
        @>....o
           \...
             \.

           /.
        @>... o
           \.
        */
        #[test]
        fn test() {
            let cases = [
                (1.0, "     O    "),
                (0.6, "     o    "),
                (0.2, "     .    "),
                (0.19, "          "),
            ];
            for (fov_range, expected) in cases {
                let food = vec![Food {
                    position: na::Point2::new(0.2, 0.5),
                }];
                TestCase {
                    fov_range,
                    fov_angle: PI / 2.0,
                    receptors: 10,
                    x: 0.0,
                    y: 0.5,
                    rotation: 0.0,
                    food,
                    expected,
                }
                .run();
            }
        }
    }

    mod test_fov_angles {
        use super::*;

        /*
            o
                o
        o   @>      o
                o
            o

            o
                o /...
        o   @>    ..o.
                o \...
            o

            o   /.....
               /o.....
        o   @>.....o..
               \o.....
            o   \.....

            o.........
            |...o.....
        o   @>.....o.
            |...o.....
            o.........

        ....o.........
        ........o.....
        o...@>.....o.
        ........o.....
        ....o.........
        */
        #[test]
        fn test() {
            let cases = [
                (0.0, "o         "),
                (PI / 180.0, "     o    "),
                (PI / 2.0, ".    o   ."),
                (PI, "o .  o . o"),
                (2.0 * PI, "  o. o.o o"),
            ];
            for (fov_angle, expected) in cases {
                let food = vec![
                    Food {
                        position: na::Point2::new(1.0, 0.5),
                    },
                    Food {
                        position: na::Point2::new(1.0, 1.0),
                    },
                    Food {
                        position: na::Point2::new(1.0, 0.0),
                    },
                    Food {
                        position: na::Point2::new(0.5, 1.0),
                    },
                    Food {
                        position: na::Point2::new(0.5, 0.0),
                    },
                    Food {
                        position: na::Point2::new(0.0, 0.5),
                    },
                ];
                TestCase {
                    fov_range: 1.0,
                    fov_angle,
                    receptors: 10,
                    x: 0.5,
                    y: 0.5,
                    rotation: 0.0,
                    food,
                    expected,
                }
                .run();
            }
        }
    }

    mod test_receptors {
        use super::*;

        /*
            |
            | o
            @>      o
            |
            |    o

            |
            | o
            @>------o
            |
            |    o

            |     /
            | o/
            @>      o
            |  \
            |    o\
        */
        #[test]
        fn test() {
            let cases = [(1, "O"), (2, "oO"), (3, "o.O")];
            for (receptors, expected) in cases {
                let food = vec![
                    Food {
                        position: na::Point2::new(0.55, 0.6),
                    },
                    Food {
                        position: na::Point2::new(1.4, 0.5),
                    },
                    Food {
                        position: na::Point2::new(0.8, 0.1),
                    },
                ];
                TestCase {
                    fov_range: 1.0,
                    fov_angle: PI,
                    receptors,
                    x: 0.5,
                    y: 0.5,
                    rotation: 0.0,
                    food,
                    expected,
                }
                .run();
            }
        }
    }

    mod test_position {
        use super::*;

        /*
              /..
            @>...
              \..
                o

              /..
            @>..o
              \..

            /....
          @>....o
            \....

                o
              /..
            @>...
              \..
        */
        #[test]
        fn test() {
            let cases = [
                (((0.5, 0.0), " ")),
                ((0.5, 0.5), "O"),
                ((0.2, 0.5), "."),
                ((0.5, 1.0), " "),
            ];
            for ((x, y), expected) in cases {
                let food = vec![Food {
                    position: na::Point2::new(0.6, 0.5),
                }];
                TestCase {
                    fov_range: 0.5,
                    fov_angle: PI / 2.0,
                    receptors: 1,
                    x,
                    y,
                    rotation: 0.0,
                    food,
                    expected,
                }
                .run();
            }
        }
    }

    mod test_rotation {
        use super::*;

        /*
            o
              /...
          o @>...o
              \...

          ..o..
          \ ^ /
          o @    o

            o
        ...\
        ..o<@    o
        .../
        */
        #[test]
        fn test() {
            let cases = [
                (0.0, "."),
                (PI / 2.0, "o"),
                (PI, "O"),
                (3.0 * PI / 2.0, " "),
            ];
            for (rotation, expected) in cases {
                let food = vec![
                    Food {
                        position: na::Point2::new(1.4, 0.5),
                    },
                    Food {
                        position: na::Point2::new(0.5, 1.0),
                    },
                    Food {
                        position: na::Point2::new(0.4, 0.5),
                    },
                ];
                TestCase {
                    fov_range: 1.0,
                    fov_angle: PI / 2.0,
                    receptors: 1,
                    x: 0.5,
                    y: 0.5,
                    rotation,
                    food,
                    expected,
                }
                .run();
            }
        }
    }
}
