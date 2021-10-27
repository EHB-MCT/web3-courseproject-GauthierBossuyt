import React, { Component } from "react";
import Sketch from "react-p5";

export default class Canvas extends Component {
  x = 50;
  y = 50;
  width = window.innerWidth;
  height = window.innerHeight;
  //delta = Math.PI * 2 /total;
  total = 300;
  factor = 1;
  radius = this.width * 1.05;
  toggle = true;

  setup = (p5, canvasParentRef) => {
    p5.createCanvas(this.width, this.height).parent(canvasParentRef);
    console.log(canvasParentRef);
  };

  mouseWheel = (event) => {
    let scroll = event._mouseWheelDeltaY;
    scroll > 0 ? (this.factor -= 1.05) : (this.factor += 1.05);
  };

  windowResized = (p5) => {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    p5.resizeCanvas(this.width, this.height);
    console.log("resize");
  };

  getVector = (p5, i) => {
    let angle = p5.map(i % this.total, 0, this.total, 0, Math.PI * 2);
    let v = p5.constructor.Vector.fromAngle(angle + Math.PI);
    v.mult(this.radius);
    return v;
  };

  checkFactor = () => {
    if (this.factor <= 1) {
      this.toggle = true;
    }
    if (this.factor >= 5) {
      this.toggle = false;
    }
  };

  draw = (p5) => {
    this.checkFactor();
    this.toggle ? (this.factor += 0.005) : (this.factor -= 0.005);
    p5.background(20);
    p5.translate(this.width / 2, this.height / 2);
    p5.noFill();
    p5.stroke(20);

    p5.circle(0, 0, this.radius * 2);
    for (let index = 0; index < this.total; index++) {
      let v = this.getVector(p5, index);
      p5.fill(20);
      p5.circle(v.x, v.y, 16);
    }
    p5.stroke(255);
    for (let i = 0; i < this.total; i++) {
      let a = this.getVector(p5, i);
      let b = this.getVector(p5, i * this.factor);
      p5.line(a.x, a.y, b.x, b.y);
    }
  };
  render() {
    return (
      <Sketch
        className="visual"
        setup={this.setup}
        draw={this.draw}
        mouseWheel={this.mouseWheel}
        windowResized={this.windowResized}
      />
    );
  }
}
