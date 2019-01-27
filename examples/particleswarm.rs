// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::prelude::*;
use argmin::solver::particleswarm::*;

use argmin_testfunctions::himmelblau;


struct Himmelblau
{

}


impl ArgminOperator for Himmelblau {
    type Parameters = Vec<f64>;
    type OperatorOutput = f64;
    type Hessian = ();

    fn apply(&self, param: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
        Ok(himmelblau(param))
    }
}


fn run() -> Result<(), Error> {


    // Define inital parameter vector
    let init_param: Vec<f64> = vec![0.1, 0.1];

    let cost_function = Himmelblau {};

    let mut visualizer = Visualizer { fg: gnuplot::Figure::new() };
    visualizer.cost_function();

    // Set up line search method
    let mut solver = ParticleSwarm::new(&cost_function, init_param)?;

    // Attach a logger
    solver.add_logger(ArgminSlogLogger::term());

    solver.set_max_iters(10);

    let mut callback = |xy: &Vec<f64>, c: f64| visualizer.iteration(xy, c);
    solver.set_iter_callback(&mut callback);

    // Run solver
    solver.run()?;

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(100));

    // Print Result
    println!("{:?}", solver.result());
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{} {}", e.as_fail(), e.backtrace());
    }
}


struct Visualizer {
    fg: gnuplot::Figure
}

// TODO: destroy window
impl Visualizer {
    fn cost_function(&mut self) {
        use gnuplot::*;

        let zw = 61;
        let zh = 61;
        let mut z1 = Vec::with_capacity((zw * zh) as usize);
        for i in 0..zh
        {
            for j in 0..zw
            {
                let y = 8.0 * (i as f64) / zh as f64 - 4.0;
                let x = 8.0 * (j as f64) / zw as f64 - 4.0;
                z1.push(himmelblau(&vec![x, y]));
            }
        }

        self.fg.axes3d()
            .set_title("Surface fg4.2", &[])
            .surface(z1.iter(), zw, zh, Some((-4.0, -4.0, 4.0, 4.0)), &[])
            // .set_x_label("X", &[])
            // .set_y_label("Y", &[])
            // .set_z_label("Z", &[])
            // .set_z_range(Fix(0.0), Fix(2000.0))
            // .set_z_ticks(Some((Fix(100.0), 1)), &[Mirror(false)], &[])
            // .set_cb_range(Fix(-1.0), Fix(1.0))
            .set_view(0.0, 0.0)
            ;

        self.fg.show();
    }

    fn iteration(&mut self, xy: &Vec<f64>, cost: f64) {
        self.fg.axes2d().points(&[xy[0]], &[xy[1]], &[]);
        self.fg.show();

        std::thread::sleep(std::time::Duration::from_secs(1));

    }
}
