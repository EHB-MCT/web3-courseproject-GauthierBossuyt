import React, { Component } from "react";
import { Link } from "react-router-dom";

class Nav extends Component {
  render() {
    return (
      <div className="nav">
        <Link>
          <h1 className="nav title" to="/">
            <Link to="/">Hello, {this.props.name}</Link>
          </h1>
        </Link>
        <Link className="nav link" to="/pausePlay">
          Pause/Play
        </Link>
        <Link className="nav link" to="/p5js">
          P5.js
        </Link>
      </div>
    );
  }
}

export default Nav;
