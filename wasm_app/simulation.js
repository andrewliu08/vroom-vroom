export class SimulationView {
  constructor(el) {
    this.el = el;
  }

  reset() {
    const pixelRatio = window.devicePixelRatio || 1;

    const size = Math.min(window.innerWidth - 500, window.innerHeight - 50);

    this.el.width = size * pixelRatio;
    this.el.height = size * pixelRatio;
    this.el.style.width = size + "px";
    this.el.style.height = size + "px";

    this.ctxt = this.el.getContext("2d");
    this.ctxt.clearRect(0, 0, this.el.width, this.el.width);
  }

  fillAnimal(x, y, rotation) {
    const ANIMAL_SIZE = 0.01;
    const ANIMAL_COLOR = "#758b9e";
    let size = ANIMAL_SIZE * this.el.width;
    let headAngle = rotation;
    let leg1Angle = rotation + (14 * Math.PI) / 18; // +140 degrees
    let leg2Angle = rotation - (14 * Math.PI) / 18; // -140 degrees

    this.ctxt.beginPath();
    this.ctxt.moveTo(
      x + Math.cos(headAngle) * size,
      y + Math.sin(headAngle) * size
    );
    this.ctxt.lineTo(
      x + Math.cos(leg1Angle) * size,
      y + Math.sin(leg1Angle) * size
    );
    this.ctxt.lineTo(
      x + Math.cos(leg2Angle) * size,
      y + Math.sin(leg2Angle) * size
    );
    this.ctxt.moveTo(
      x + Math.cos(headAngle) * size,
      y + Math.sin(headAngle) * size
    );
    this.ctxt.fillStyle = ANIMAL_COLOR;
    this.ctxt.fill();
  }

  drawAnimals(animals) {
    for (const animal of animals) {
      this.fillAnimal(
        animal.x * this.el.width,
        animal.y * this.el.height,
        animal.rotation
      );
    }
  }

  fillFood(x, y) {
    const FOOD_SIZE = 0.003;
    const FOOD_COLOR = "#b4a794";
    let size = FOOD_SIZE * this.el.width;
    this.ctxt.beginPath();
    this.ctxt.arc(x, y, size, 0, 2 * Math.PI);
    this.ctxt.fillStyle = FOOD_COLOR;
    this.ctxt.fill();
  }

  drawFood(food) {
    for (const f of food) {
      this.fillFood(f.x * this.el.width, f.y * this.el.height);
    }
  }
}
