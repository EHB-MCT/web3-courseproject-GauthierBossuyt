import React from "react";
import ReactDOM from "react-dom";
import "./index.css";

/*COMPONENTS*/
import Canvas from "./sketch";

ReactDOM.render(
  <React.StrictMode>
    <div id="card">
      <Canvas></Canvas>
    </div>
  </React.StrictMode>,
  document.getElementById("root")
);
