import React from "react";
import Lottie from "react-lottie";
import animation from "../animations/9844-loading-40-paperplane.json";

export default class PausePlay extends React.Component {
  state = { isPaused: false };

  buttonLabel() {
    return this.state.isPaused ? "Resume" : "Pause";
  }

  render() {
    const defaultOptions = {
      loop: true,
      autoplay: true,
      animationData: animation,
      rendererSettings: {
        preserveAspectRatio: "xMidYMid slice",
      },
    };
    return (
      <div className="animation">
        <Lottie
          options={defaultOptions}
          isPaused={this.state.isPaused}
        ></Lottie>

        <button
          onClick={() => this.setState({ isPaused: !this.state.isPaused })}
          className="button"
        >
          {this.buttonLabel()}
        </button>
      </div>
    );
  }
}
