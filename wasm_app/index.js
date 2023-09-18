import * as sim from "lib_simulation_wasm";

const simulation = new sim.Simulation();

const viewport = document.getElementById("viewport");
const ctx = viewport.getContext("2d");

const viewportScale = window.devicePixelRatio || 1;
viewport.width = viewport.clientWidth * viewportScale;
viewport.height = viewport.clientHeight * viewportScale;
viewport.style.width = viewport.clientWidth + "px";
viewport.style.height = viewport.clientHeight + "px";

const ANIMAL_SIZE = 0.015;
const FOOD_SIZE = 0.005;
const ANIMAL_COLOR = "gray";
const FOOD_COLOR = "green";

CanvasRenderingContext2D.prototype.fillAnimal = function (x, y, rotation) {
  let size = ANIMAL_SIZE * viewport.width;
  let headAngle = rotation;
  let leg1Angle = rotation + (14 * Math.PI) / 18; // +140 degrees
  let leg2Angle = rotation - (14 * Math.PI) / 18; // -140 degrees

  this.beginPath();
  this.moveTo(x + Math.cos(headAngle) * size, y + Math.sin(headAngle) * size);
  this.lineTo(x + Math.cos(leg1Angle) * size, y + Math.sin(leg1Angle) * size);
  this.lineTo(x + Math.cos(leg2Angle) * size, y + Math.sin(leg2Angle) * size);
  this.moveTo(x + Math.cos(headAngle) * size, y + Math.sin(headAngle) * size);
  this.fillStyle = ANIMAL_COLOR;
  this.fill();
};

CanvasRenderingContext2D.prototype.fillFood = function (x, y) {
  let size = FOOD_SIZE * viewport.width;
  this.beginPath();
  this.arc(x, y, size, 0, 2 * Math.PI);
  this.fillStyle = FOOD_COLOR;
  this.fill();
};

function draw_animals(simulation) {
  for (const animal of simulation.world().animals) {
    ctx.fillAnimal(
      animal.x * viewport.width,
      animal.y * viewport.height,
      animal.rotation
    );
  }
}

function draw_food(simulation) {
  for (const food of simulation.world().food) {
    ctx.fillFood(food.x * viewport.width, food.y * viewport.height);
  }
}

function redraw() {
  ctx.clearRect(0, 0, viewport.width, viewport.height);
  simulation.step();

  draw_animals(simulation);
  draw_food(simulation);
  requestAnimationFrame(redraw);
}

draw_animals(simulation);
draw_food(simulation);
redraw();
