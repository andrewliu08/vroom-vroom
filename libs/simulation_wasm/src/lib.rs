use rand::{rngs::ThreadRng, thread_rng};
use serde::Serialize;
use serde_wasm_bindgen::to_value;
use wasm_bindgen::prelude::*;

use lib_simulation as sim;

#[wasm_bindgen]
pub struct Simulation {
    rng: ThreadRng,
    sim: sim::Simulation,
}

#[derive(Clone, Debug, Serialize)]
pub struct World {
    animals: Vec<Animal>,
    food: Vec<Food>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Animal {
    x: f64,
    y: f64,
    rotation: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Food {
    x: f64,
    y: f64,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut rng = thread_rng();
        let num_animals = 5;
        let num_food = 100;
        let sim = sim::Simulation::random(&mut rng, num_animals, num_food);
        Self { rng, sim }
    }

    pub fn world(&self) -> JsValue {
        let world = World::from(self.sim.world());
        to_value(&world).unwrap()
    }

    pub fn generation(&self) -> u32 {
        self.sim.generation()
    }

    pub fn generation_steps(&self) -> u32 {
        self.sim.generation_steps()
    }

    pub fn step(&mut self) {
        self.sim.step(&mut self.rng);
    }
}

impl From<&sim::World> for World {
    fn from(world: &sim::World) -> Self {
        let animals = world.animals().iter().map(Animal::from).collect();
        let food = world.food().iter().map(Food::from).collect();
        Self { animals, food }
    }
}

impl From<&sim::Animal> for Animal {
    fn from(animal: &sim::Animal) -> Self {
        Self {
            x: animal.position().x,
            y: animal.position().y,
            rotation: animal.rotation().angle(),
        }
    }
}

impl From<&sim::Food> for Food {
    fn from(food: &sim::Food) -> Self {
        Self {
            x: food.position().x,
            y: food.position().y,
        }
    }
}
