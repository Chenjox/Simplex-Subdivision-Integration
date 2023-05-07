use indextree::{Arena, NodeEdge, NodeId};
use ndarray::prelude::*;
use std::mem;
use std::{fmt::write};
use std::fs::File;
use std::io::Write;

use crate::{
    integration_2d::{
        functions::{Constant2DFunction, Function2DHistory},
        integrators::{
            quadrilaterial_integrator::*, Hierarchic2DIntegrator, Hierarchic2DIntegratorData,
        },
        *,
    },
    problems::PhaseField2DFunction,
};

mod integration_2d;
mod problems;

fn precision_test(precision: f64) {
    let sim = Simplex2D::new_from_points(
        &array![1., 1.],
        &array![1.5, 1. + (3.0f64).sqrt() / 2.],
        &array![2., 1.],
    );
    let inte1 = Quadrilateral2DIntegrator::new(1);
    let inte2 = Hierarchic2DIntegrator::new(inte1, false, precision);
    //let inte1 = Quadrilateral2DIntegrator::new(3);

    let func = Box::new(Function2DHistory::new(PhaseField2DFunction {
        weights: [10.0, 10.0, 10.0, 10.0, -0., -0.],
    }));
    //let func = Box::new(Function2DHistory::new(Constant2DFunction));

    let mut cache = Hierarchic2DIntegratorData::new_cache();

    //let result1 = inte1.integrate_simplex(&func, &sim, &mut IntegratorDummy::get());
    let result2 = inte2.integrate_simplex(&func, &sim, &mut cache);

    let evals_build = func.function_evaluations();
    func.delete_history();

    let result = inte2.integrate_simplex(&func, &sim, &mut cache);

    let hist = func.get_history();
    let hist_size = mem::size_of_val(&*hist) as f64/(1024.0*1024.0);

    println!("{},{},{},{},{} MB",  result,precision, evals_build, hist.len(),hist_size);
}

fn main() {
    // ASSERTION: The Simplex is always rightly oriented.
    for i in 0..13 {
        let prec = 1.0 / (10.0f64).powi(i);
        precision_test(prec);
    }

    //for el in &hist {
    //    //println!("{},{}",el, el.fold(0., |f1, f2| f1 + f2));
    //    println!("\\draw[fill,red] (barycentric cs:ca={:.3},cb={:.3},cc={:.3}) coordinate (cb1) circle (2pt);",el[0],el[1],el[2]);
    //}
    /*
    let mut f = File::create("output.csv").expect("Unable to create file");
    let points = sim.get_points();
    for i in 0..3 {
        writeln!(f, "{} {} 0.0", points[[0, i]], points[[1, i]]).expect("Unable to write!");
    }
    for i in &hist {
        let p = points.dot(&i.0);
        writeln!(f, "{} {} {}", p[0], p[1], i.1).expect("Unable to write!");
    }*/

}
