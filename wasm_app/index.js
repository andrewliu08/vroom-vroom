import * as sim from "lib_simulation_wasm";
import { SimulationView } from "./simulation";

const simulation = new sim.Simulation();

const simulationView = new SimulationView(
  document.getElementById("simulation")
);

function setControllerText(text) {
  const controllerTextElement = document.getElementById("controllerText");
  const formattedText = text.replace(/\n/g, "<br>");
  controllerTextElement.innerHTML = formattedText;
}

function redraw() {
  simulationView.reset();
  for (let i = 0; i < 1; i++) {
    simulation.step();
  }
  let stats = simulation.prev_generation_statistics();
  let max_fitness = null;
  let min_fitness = null;
  let mean_fitness = null;
  let std_fitness = null;
  if (stats) {
    max_fitness = stats.max_fitness;
    min_fitness = stats.min_fitness;
    mean_fitness = stats.mean_fitness;
    std_fitness = stats.std_fitness;
  }

  simulationView.drawAnimals(simulation.world().animals);
  simulationView.drawFood(simulation.world().food);
  requestAnimationFrame(redraw);

  let text = `Generation: ${simulation.generation()}\n`;
  text += `Generation steps: ${simulation.generation_steps()}\n\n`;
  text += `Prev generation stats:\n`;
  text += `Max fitness: ${max_fitness}\n`;
  text += `Min fitness: ${min_fitness}\n`;
  text += `Mean fitness: ${mean_fitness}\n`;
  text += `Std fitness: ${std_fitness}\n`;
  setControllerText(text);
}

redraw();
