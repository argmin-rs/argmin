// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::prelude::*;
use argmin::solver::particleswarm::*;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

use argmin_testfunctions::himmelblau;

#[derive(Default, Clone, Serialize, Deserialize)]
struct Himmelblau {}

impl ArgminOp for Himmelblau {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(himmelblau(param))
    }
}

fn run() -> Result<(), Error> {
    // Define inital parameter vector
    let init_param: Vec<f64> = vec![0.1, 0.1];

    let cost_function = Himmelblau {};

    let visualizer = Visualizer3d::new()
        .delay(std::time::Duration::from_secs(1))
        .surface(Surface::new::<Himmelblau>(
            cost_function.clone(),
            (-4.0, -4.0, 4.0, 4.0),
            0.1,
        ));

    {
        let solver = ParticleSwarm::new((vec![-4.0, -4.0], vec![4.0, 4.0]), 100, 0.5, 0.0, 0.5)?;

        let res = Executor::new(cost_function, solver, init_param)
            .add_observer(visualizer, ObserverMode::Always)
            .max_iters(15)
            .run()?;

        // Wait a second (lets the logger flush everything before printing again)
        std::thread::sleep(std::time::Duration::from_secs(100));

        // Print Result
        println!("{}", res);
    }

    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{} {}", e.as_fail(), e.backtrace());
    }
}

/// Helper class for visualized surface
struct Surface {
    window: (f64, f64, f64, f64),
    width: usize,
    height: usize,
    zvalues: Vec<f64>,
}

impl Surface {
    fn new<O>(op: O, window: (f64, f64, f64, f64), resolution: f64) -> Self
    where
        O: ArgminOp<Param = Vec<f64>, Output = f64>,
    {
        let width = window.2 - window.0;
        let height = window.3 - window.1;
        let num_x = (width / resolution) as usize;
        let num_y = (height / resolution) as usize;

        let mut zvalues: Vec<f64> = vec![];

        for i in 0..num_y {
            for j in 0..num_x {
                let y = height * (i as f64) / num_y as f64 - (0.5 * height);
                let x = width * (j as f64) / num_x as f64 - (0.5 * width);
                if let Ok(zvalue) = op.apply(&vec![x, y]) {
                    zvalues.push(zvalue);
                }
            }
        }

        Self {
            window: window,
            width: num_x,
            height: num_y,
            zvalues,
        }
    }
}

/// Visualize iterations of a solver for cost functions of type
/// (x,y) -> cost
/// , where x and y are real numbers. If the solver is population-based,
/// The current population is also visualized.
struct Visualizer3d {
    // Need mutex because `Figure` contains `Cell`
    fg: Mutex<gnuplot::Figure>,
    optima_x: Vec<f64>,
    optima_y: Vec<f64>,
    optima_z: Vec<f64>,
    particles_x: Vec<f64>,
    particles_y: Vec<f64>,
    particles_z: Vec<f64>,
    /// Optional visualized surface of cost function
    surface: Option<Surface>,
    /// Optional delay between iterations
    delay: Option<std::time::Duration>,
}

impl Visualizer3d {
    fn new() -> Self {
        Self {
            fg: Mutex::new(gnuplot::Figure::new()),
            optima_x: vec![],
            optima_y: vec![],
            optima_z: vec![],
            particles_x: vec![],
            particles_y: vec![],
            particles_z: vec![],
            surface: None,
            delay: None,
        }
    }

    fn delay(mut self, duration: std::time::Duration) -> Self {
        self.delay = Some(duration);

        self
    }

    fn surface(mut self, surface: Surface) -> Self {
        self.surface = Some(surface);

        self
    }

    fn draw(&mut self) {
        use gnuplot::*;

        // TODO: unwrap evil
        let mut figure = self.fg.lock().unwrap();

        figure.clear_axes();

        let options_optima = [Color("#ffff00"), PointSize(2.0)];
        let options_particles = [Color("#ff0000"), PointSize(2.0)];
        let axes3d = figure.axes3d();

        // Draw surface before points
        if let Some(surface) = &self.surface {
            let window = Some(surface.window);
            axes3d.surface(
                surface.zvalues.iter(),
                surface.width,
                surface.height,
                window,
                &[],
            );
        }

        axes3d
            .points(
                &self.optima_x,
                &self.optima_y,
                &self.optima_z,
                &options_optima,
            )
            .points(
                &self.particles_x,
                &self.particles_y,
                &self.particles_z,
                &options_particles,
            )
            .set_view(30.0, 30.0); // TODO: do not reset view on new iteration

        figure.show();

        if let Some(delay) = self.delay {
            std::thread::sleep(delay);
        };
    }

    fn iteration(&mut self, xy: &Vec<f64>, cost: f64, population: Option<&Vec<(Vec<f64>, f64)>>) {
        self.optima_x.clear();
        self.optima_y.clear();
        self.optima_z.clear();
        self.optima_x.push(xy[0]);
        self.optima_y.push(xy[1]);
        self.optima_z.push(cost);

        self.particles_x.clear();
        self.particles_y.clear();
        self.particles_z.clear();

        if let Some(population) = population {
            for (param, cost) in population {
                self.particles_x.push(param[0]);
                self.particles_y.push(param[1]);
                self.particles_z.push(*cost);
            }
        }

        self.draw();
    }
}

impl<O> Observe<O> for Visualizer3d
where
    O: ArgminOp<Param = Vec<f64>>,
{
    fn observe_iter(&mut self, state: &IterState<O>, _kv: &ArgminKV) -> Result<(), Error> {
        // TODO: get particles from `state` or `kv`

        self.iteration(&state.param, state.best_cost, state.get_population());

        Ok(())
    }
}
