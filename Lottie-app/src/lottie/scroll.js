import React from "react";
import "@lottiefiles/lottie-player";
import { create } from "@lottiefiles/lottie-interactivity";

class Scroll extends React.Component {
  constructor(props) {
    super(props);
    this.myRef = React.createRef();
  }
  componentDidMount() {
    this.myRef.current.addEventListener("load", function (e) {
      create({
        mode: "scroll",
        player: "#superhot",
        actions: [
          {
            visibility: [0, 0.6],
            type: "stop",
            frames: [100],
          },
          {
            visibility: [0.6, 0.8],
            type: "seek",
            frames: [30, 120],
          },
          {
            visibility: [0.8, 1],
            type: "stop",
            frames: [120],
          },
        ],
      });
    });
  }
  render() {
    return (
      <div className="scroll">
        <lottie-player
          ref={this.myRef}
          id="superhot"
          src="booting.json"
          style={{ margin_top: "300px", height: "50vh" }}
        ></lottie-player>
        <div style={{ height: "60vh" }}>
          <h1></h1>
        </div>
      </div>
    );
  }
}

export default Scroll;
