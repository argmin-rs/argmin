// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Observer which visualizes the progress of the solver

extern crate gnuplot;
use crate::core::{ArgminFloat, ArgminKV, ArgminOp, Error, IterState, Observe};
use instant;
use std::sync::Mutex;

/// Visualize iterations of a solver for cost functions of type
/// (x,y) -> cost
/// , where x and y are real numbers. If the solver is population-based,
/// The current population is also visualized.
pub struct Visualizer3d {
    // Need mutex because `Figure` contains `Cell`
    /// Figure handle
    fg: Mutex<gnuplot::Figure>,
    /// x component of optima
    optima_x: Vec<f64>,
    /// y component of optima
    optima_y: Vec<f64>,
    /// z component of optima
    optima_z: Vec<f64>,
    /// x component of particles
    particles_x: Vec<f64>,
    /// y component of particles
    particles_y: Vec<f64>,
    /// z component of particles
    particles_z: Vec<f64>,
    /// Optional visualized surface of cost function
    surface: Option<Surface>,
    /// Optional delay between iterations
    delay: Option<instant::Duration>,
}

impl Visualizer3d {
    /// Create a new visualizer
    pub fn new() -> Self {
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

    /// Set delay
    #[must_use]
    pub fn delay(mut self, duration: instant::Duration) -> Self {
        self.delay = Some(duration);
        self
    }

    /// Set surface
    #[must_use]
    pub fn surface(mut self, surface: Surface) -> Self {
        self.surface = Some(surface);
        self
    }

    /// Draw
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

        // TODO: unwrap evil
        figure.show().unwrap();

        if let Some(delay) = self.delay {
            std::thread::sleep(delay);
        };
    }

    /// TODO
    fn iteration<F: ArgminFloat>(
        &mut self,
        xy: &[F],
        cost: F,
        population: Option<&Vec<(Vec<F>, F)>>,
    ) {
        self.optima_x.clear();
        self.optima_y.clear();
        self.optima_z.clear();
        self.optima_x.push(F::to_f64(&xy[0]).unwrap());
        self.optima_y.push(F::to_f64(&xy[1]).unwrap());
        self.optima_z.push(F::to_f64(&cost).unwrap());

        self.particles_x.clear();
        self.particles_y.clear();
        self.particles_z.clear();

        if let Some(population) = population {
            for (param, cost) in population {
                self.particles_x.push(F::to_f64(&param[0]).unwrap());
                self.particles_y.push(F::to_f64(&param[1]).unwrap());
                self.particles_z.push(F::to_f64(cost).unwrap());
            }
        }

        self.draw();
    }
}

impl std::default::Default for Visualizer3d {
    fn default() -> Self {
        Visualizer3d::new()
    }
}

impl<O> Observe<O> for Visualizer3d
where
    O: ArgminOp<Param = Vec<f64>, Float = f64>,
{
    fn observe_iter(&mut self, state: &IterState<O>, _kv: &ArgminKV) -> Result<(), Error> {
        // TODO: get particles from `state` or `kv`

        self.iteration(&state.param, state.best_cost, state.get_population());

        Ok(())
    }
}

/// Helper class for visualized surface
pub struct Surface {
    /// Window dimenstions
    window: (f64, f64, f64, f64),
    /// Width of surface
    width: usize,
    /// Height of surface
    height: usize,
    /// TODO
    zvalues: Vec<f64>,
}

impl Surface {
    /// Create a new surface
    pub fn new<O>(op: O, window: (f64, f64, f64, f64), resolution: f64) -> Self
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
            window,
            width: num_x,
            height: num_y,
            zvalues,
        }
    }
}
