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

type Particles = Vec<Particle<Vec<f64>>>;

fn run() -> Result<(), Error> {
    // Define inital parameter vector
    let init_param: Vec<f64> = vec![0.1, 0.1];

    let cost_function = Himmelblau {};

    let visualizer = ParticleSwarmVisualizer::new();

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
    fn new(window: (f64, f64, f64, f64), resolution: f64) -> Self {
        let width = window.2 - window.0;
        let height = window.3 - window.1;
        let num_x = (width / resolution) as usize;
        let num_y = (height / resolution) as usize;

        let mut zvalues: Vec<f64> = vec![];

        for i in 0..num_y {
            for j in 0..num_x {
                let y = height * (i as f64) / num_y as f64 - (0.5 * height);
                let x = width * (j as f64) / num_x as f64 - (0.5 * width);
                zvalues.push(himmelblau(&vec![x, y]));
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

struct ParticleSwarmVisualizer {
    // Need mutex because `Figure` contains `Cell`
    fg: Mutex<gnuplot::Figure>,
    optima_x: Vec<f64>,
    optima_y: Vec<f64>,
    optima_z: Vec<f64>,
    particles_x: Vec<f64>,
    particles_y: Vec<f64>,
    particles_z: Vec<f64>,

    surface: Surface,
}

// TODO: destroy window
impl ParticleSwarmVisualizer {
    fn new() -> Self {
        Self {
            fg: Mutex::new(gnuplot::Figure::new()),
            optima_x: vec![],
            optima_y: vec![],
            optima_z: vec![],
            particles_x: vec![],
            particles_y: vec![],
            particles_z: vec![],
            surface: Surface::new((-4.0, -4.0, 4.0, 4.0), 0.1),
        }
    }

    fn draw(&mut self) {
        use gnuplot::*;

        // TODO: unwrap evil
        let mut figure = self.fg.lock().unwrap();

        figure.clear_axes();

        let options_optima = [Color("#ffff00"), PointSize(2.0)];
        let options_particles = [Color("#ff0000"), PointSize(2.0)];
        let window = Some(self.surface.window);
        figure
            .axes3d()
            .surface(
                self.surface.zvalues.iter(),
                self.surface.width,
                self.surface.height,
                window,
                &[],
            )
            .points(
                &self.optima_x,
                &self.optima_y,
                &self.optima_z,
                &options_optima,
            )
            // .points(
            //     &self.particles_x,
            //     &self.particles_y,
            //     &self.particles_z,
            //     &options_particles,
            // )
            .set_view(0., 0.);
        figure.show();

        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    fn iteration(&mut self, xy: &Vec<f64>, cost: f64, particles: &Particles) {
        self.optima_x.clear();
        self.optima_y.clear();
        self.optima_z.clear();
        self.optima_x.push(xy[0]);
        self.optima_y.push(xy[1]);
        self.optima_z.push(cost);

        self.particles_x.clear();
        self.particles_y.clear();
        self.particles_z.clear();
        for particle in particles {
            self.particles_x.push(particle.position[0]);
            self.particles_y.push(particle.position[1]);
            self.particles_z.push(particle.cost);
        }

        self.draw();
    }
}

impl<O> Observe<O> for ParticleSwarmVisualizer
where
    O: ArgminOp<Param = Vec<f64>>,
{
    fn observe_iter(&mut self, state: &IterState<O>, _kv: &ArgminKV) -> Result<(), Error> {
        // TODO: get particles from `state` or `kv`

        self.iteration(&state.param, state.best_cost, &vec![]);

        Ok(())
    }
}
