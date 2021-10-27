import React from "react";
import ReactDOM from "react-dom";
import "./index.scss";
import reportWebVitals from "./reportWebVitals";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";

import Nav from "./navigation/nav";
import PausePlay from "./lottie/pausePlay";
import Scroll from "./lottie/scroll";
import P5js from "./p5js/p5js";

ReactDOM.render(
  <React.StrictMode>
    <Router>
      <Nav name="Gauthier"></Nav>
      <div className="content">
        <Switch>
          <Route path="/pausePlay">
            <PausePlay></PausePlay>
          </Route>
          <Route path="/p5js">
            <P5js></P5js>
          </Route>
          <Route path="/">
            <Scroll></Scroll>
          </Route>
        </Switch>
      </div>
    </Router>
  </React.StrictMode>,
  document.getElementById("root")
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals(console.log);
