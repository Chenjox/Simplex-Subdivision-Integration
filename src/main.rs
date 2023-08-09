use integration_2d::functions::RepeatedPyramidFunction;
use integration_3d::{
    functions::{Constant3DFunction, Function3DHistory},
    integrators::Quadrilateral3DIntegrator,
    Simplex3D, Simplex3DIntegrator,
};
use ndarray::prelude::*;
use std::fs::File;
use std::io::Write;

use crate::{
    integration_2d::{
        functions::{Constant2DFunction, Function2DHistory, PyramidFunction},
        integrators::{
            quadrilaterial_integrator::*, Hierarchic2DIntegrator, Hierarchic2DIntegratorData,
        },
        *,
    },
    problems::PhaseField2DFunction, integration_3d::{integrators::{Hierarchic3DIntegrator, Hierarchic3DIntegratorData}, functions::Multiplicative3DFunction},
};

mod integration_2d;
mod integration_3d;
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

    println!("{},{},{},{}", result, precision, evals_build, hist.len());
}

fn integration_test() {
    let sim = Simplex2D::new_from_points(
        &array![1., 1.],
        &array![1.5, 1. + (3.0f64).sqrt() / 2.],
        &array![2., 1.],
    );
    let precision = 1e-7;
    let inte1 = Quadrilateral2DIntegrator::new(1);
    let inte2 = Hierarchic2DIntegrator::new(inte1, false, precision);
    let func = Box::new(Function2DHistory::new(PhaseField2DFunction {
        weights: [10.0, 10.0, 10.0, 10.0, -0., -0.],
    }));
    let func2 = Box::new(Function2DHistory::new(RepeatedPyramidFunction::new(vec![
        [0.2, 0.3, 0.5, 3.0],
        [0.5, 0.3, 0.2, 3.0],
    ])));

    let mut cache = Hierarchic2DIntegratorData::new_cache();

    //let result1 = inte1.integrate_simplex(&func, &sim, &mut IntegratorDummy::get());
    let result2 = inte2.integrate_simplex(&func, &sim, &mut cache);

    let evals_build = func.function_evaluations();
    func.delete_history();

    let result = inte2.integrate_simplex(&func2, &sim, &mut cache);

    let hist = func2.get_history();

    let mut f = File::create("output.csv").expect("Unable to create file");
    let points = sim.get_points();
    for i in 0..3 {
        writeln!(f, "{} {} 0.0", points[[0, i]], points[[1, i]]).expect("Unable to write!");
    }
    for i in &hist {
        let p = points.dot(&i.0);
        writeln!(f, "{} {} {}", p[0], p[1], i.1).expect("Unable to write!");
    }
}

fn integration_testing() {
    let sim = Simplex3D::new_from_points(
        &array![0.,0.,0.], 
        &array![1.,0.,0.],
        &array![0.,1.,0.],
        &array![0.,0.,1.],);

    let func = Box::new(Function3DHistory::new(Constant3DFunction {}));

    let inte = Quadrilateral3DIntegrator::new(2);
    let hierarchic_inte = Hierarchic3DIntegrator::new(inte, true, 1e-5);
    let inte = hierarchic_inte;

    //let hierarchy = vec![0, 20, 1, 20,1,2,3,4,13,21, 2, 3, 4, 13,21];
    let hierarchy = vec![0, 20, 1, 2, 3, 4, 13,20,5,6,7,8,9,10,11,12,14,15,16,17,18,19,21,21];
    let mut cache = Hierarchic3DIntegratorData::new_cache_from_vec_tree(&hierarchy);


    let result = inte.integrate_simplex(&func, &sim, &mut cache);

    let hist = func.get_history();

    println!("{}", hist.len());
    let mut file = File::create(&"out.csv").unwrap();
    for el in &hist {
        //println!("{},{}",el, el.fold(0., |f1, f2| f1 + f2));
        let sim_points = sim.get_points();
        let el = &el.0;
        let point = sim_points.dot(el);
        write!(file, "{} {} {}\n",point[0], point[1], point[2]).unwrap();
        //println!(
        //    "\\draw[fill,red] (barycentric cs:b1={:.3},b2={:.3},b3={:.3},b4={:.3}) circle (2pt);",
        //    el[0], el[1], el[2], el[3]
        //);
    }

    println!("{}",result);

}

fn main() {
    // ASSERTION: The Simplex is always rightly oriented.
    //for i in 0..13 {
    //    let prec = 1.0 / (10.0f64).powi(i);
    //    precision_test(prec);
    //}
    //for el in &hist {
    //    //println!("{},{}",el, el.fold(0., |f1, f2| f1 + f2));
    //    println!("\\draw[fill,red] (barycentric cs:ca={:.3},cb={:.3},cc={:.3}) coordinate (cb1) circle (2pt);",el[0],el[1],el[2]);
    //}
    //integration_testing();
    
    let sim = Simplex3D::new_from_points(
        &array![(8.0f64 / 9.0).sqrt(), 0., -1.0 / 3.0],
        &array![-(2.0f64 / 9.0).sqrt(), (2.0f64 / 3.0).sqrt(), -1.0 / 3.0],
        &array![-(2.0f64 / 9.0).sqrt(), -(2.0f64 / 3.0).sqrt(), -1.0 / 3.0],
        &array![0.0, 0.0, 1.0],
    );

    let sim = Simplex3D::new_from_points(
        &array![0.,0.,0.], 
        &array![1.,0.,0.],
        &array![0.,1.,0.],
        &array![0.,0.,1.],);

    let func = Box::new(Function3DHistory::new(Multiplicative3DFunction {}));

    let inte = Quadrilateral3DIntegrator::new(2);
    let hierarchic_inte = Hierarchic3DIntegrator::new(inte, false, 1e-5);
    let inte = hierarchic_inte;

    let mut cache = Hierarchic3DIntegratorData::new_cache();
    let result = inte.integrate_simplex(&func, &sim, &mut cache);

    let hist = func.get_history();

    println!("{}", hist.len());
    let mut file = File::create(&"out.csv").unwrap();
    for el in &hist {
        //println!("{},{}",el, el.fold(0., |f1, f2| f1 + f2));
        let sim_points = sim.get_points();
        let el = &el.0;
        let point = sim_points.dot(el);
        write!(file, "{} {} {}\n",point[0], point[1], point[2]).unwrap();
        //println!(
        //    "\\draw[fill,red] (barycentric cs:b1={:.3},b2={:.3},b3={:.3},b4={:.3}) circle (2pt);",
        //    el[0], el[1], el[2], el[3]
        //);
    }
    println!("{},{}", result, sim.get_volume());
    
}
