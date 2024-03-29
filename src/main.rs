use crate::common::IntegratorDummy;
use integration_2d::{
    functions::RepeatedPyramidFunction,
    integrators::{hierarchic_integrator, DunavantIntegrator, EdgeSubdivisionIntegrator},
};
use integration_3d::{
    functions::{Constant3DFunction, Function3DHistory},
    integrators::{
        edge_subdivision_integrator::EdgeSubdivisionIntegrator as Edge3D, Quadrilateral3DIntegrator,
    },
    Simplex3D, Simplex3DIntegrator,
};
use integration_tests::{create_figures, edge_refinement_test_2d};
use ndarray::prelude::*;
use std::io::Write;
use std::{fs::File, time::Instant};

use crate::{
    integration_2d::{
        functions::{Constant2DFunction, Function2DHistory, PyramidFunction},
        integrators::{
            quadrilaterial_integrator::*, Hierarchic2DIntegrator, Hierarchic2DIntegratorData,
        },
        *,
    },
    integration_3d::{
        functions::Multiplicative3DFunction,
        integrators::{Hierarchic3DIntegrator, Hierarchic3DIntegratorData},
    },
    problems::{
        problem_definition::{
            problem_2d_definition::{PhaseFieldFuncDiff22D, PhaseFieldFuncMatrix2D},
            problem_3d_definition::PhaseFieldFuncDiff23D,
        },
        PhaseField2DFunction,
    },
};

mod common;
mod integration_2d;
mod integration_3d;
mod integration_tests;
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

    println!(
        "{},{},{},{}",
        result.get(),
        precision,
        evals_build,
        hist.len()
    );
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
        let p = points.dot(i);
        writeln!(f, "{} {}", p[0], p[1]).expect("Unable to write!");
    }
}

fn integration_testing() {
    let sim = Simplex3D::new_from_points(
        &array![0., 0., 0.],
        &array![1., 0., 0.],
        &array![0., 1., 0.],
        &array![0., 0., 1.],
    );

    let func = Box::new(Function3DHistory::new(Constant3DFunction {}));

    let inte = Quadrilateral3DIntegrator::new(2);
    let hierarchic_inte = Hierarchic3DIntegrator::new(inte, true, 1e-5);
    let inte = hierarchic_inte;

    //let hierarchy = vec![0, 20, 1, 20,1,2,3,4,13,21, 2, 3, 4, 13,21];
    let hierarchy = vec![
        0, 20, 1, 2, 3, 4, 13, 20, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 21, 21,
    ];
    let mut cache = Hierarchic3DIntegratorData::new_cache_from_vec_tree(&hierarchy);

    let result = inte.integrate_simplex(&func, &sim, &mut cache);

    let hist = func.get_history();

    println!("{}", hist.len());
    let mut file = File::create(&"out.csv").unwrap();
    for el in &hist {
        //println!("{},{}",el, el.fold(0., |f1, f2| f1 + f2));
        let sim_points = sim.get_points();
        let el = el;
        let point = sim_points.dot(el);
        write!(file, "{} {} {}\n", point[0], point[1], point[2]).unwrap();
        //println!(
        //    "\\draw[fill,red] (barycentric cs:b1={:.3},b2={:.3},b3={:.3},b4={:.3}) circle (2pt);",
        //    el[0], el[1], el[2], el[3]
        //);
    }

    println!("{}", result.get());
}

fn integration_3d_testing() {
    let sim = Simplex3D::new_from_points(
        &array![0., 0., 0.],
        &array![1., 0., 0.],
        &array![0., 1., 0.],
        &array![0., 0., 1.],
    );

    let func = Box::new(Function3DHistory::new(Multiplicative3DFunction {}));

    let inte = Quadrilateral3DIntegrator::new(3);
    let hierarchic_inte = Hierarchic3DIntegrator::new(inte, false, 1e-5);
    let inte = hierarchic_inte;

    let mut cache = Hierarchic3DIntegratorData::new_cache();
    let result = inte.integrate_simplex(&func, &sim, &mut cache);

    func.delete_history();

    let result = inte.integrate_simplex(&func, &sim, &mut cache);

    let hist = func.get_history();

    println!("{},{}", hist.len(), result.get_borrow());
    //let mut file = File::create(&"out.csv").unwrap();
    //for el in &hist {
    //    //println!("{},{}",el, el.fold(0., |f1, f2| f1 + f2));
    //    let sim_points = sim.get_points();
    //    let el = &el.0;
    //    let point = sim_points.dot(el);
    //    write!(file, "{} {} {}\n",point[0], point[1], point[2]).unwrap();
    //    //println!(
    //    //    "\\draw[fill,red] (barycentric cs:b1={:.3},b2={:.3},b3={:.3},b4={:.3}) circle (2pt);",
    //    //    el[0], el[1], el[2], el[3]
    //    //);
    //}
    //println!("{},{}", result, sim.get_volume());
}

fn get_diagonal_order(highest_index: usize) -> Vec<(usize, usize, usize)> {
    let mut res_vec = Vec::new();
    {
        let col = highest_index;
        let mut start = 0;
        let mut offset = 0;
        let mut switcher = false;
        let mut counter = 0;
        loop {
            if switcher {
                res_vec.push((counter, start, start - offset));
            } else {
                res_vec.push((counter, start - offset, start));
            }
            counter += 1;
            start += 1;
            if start > col {
                if switcher || offset == 0 {
                    offset += 1;
                    switcher = false;
                } else {
                    switcher = true;
                }
                start = offset;
            }
            if offset > col {
                break;
            }
        }
    }
    return res_vec;
}

fn matrix_integration_test_2d() {
    let sim = Simplex2D::new_from_points(
        &array![1., 1.],
        &array![2., 1.],
        &array![1.5, 1. + (3.0f64).sqrt() / 2.],
    );

    let mut res = Array2::<f64>::zeros([6, 6]);
    let nodal_values = array![1.0, 1.0, 1.0, -1.0, 0.0, 0.0];

    let basic_integrator = DunavantIntegrator::new(2);
    let hierarchic_inte = Hierarchic2DIntegrator::new(basic_integrator, false, 1e-3);

    //let hierarchic_inte = EdgeSubdivisionIntegrator::new(basic_integrator, 100);
    let mut cache = Hierarchic2DIntegratorData::new_cache();

    // Zuerst die Hauptdiagonale, dann die Nebendiagonalen
    let res_vec = get_diagonal_order(5);
    //println!("{:?}",res_vec);

    for (count, i, j) in res_vec {
        let func = Box::new(Function2DHistory::new(PhaseFieldFuncDiff22D::new(
            nodal_values.clone(),
            1e-6,
            1.,
            i,
            j,
        )));
        let now = Instant::now();
        res[[i, j]] = hierarchic_inte
            .integrate_simplex(&func, &sim, &mut cache)
            .get(); //&mut cache
        let elapsed_time = now.elapsed();
        if count < 6 + 1 * 5 {
            // Wenn Diagonale und erste nebendiagonale durch sind
            cache.make_leafs_unchecked();
        }
        println!(
            "Running [{},{}] took {} milliseconds. Integration Points are: {}",
            i,
            j,
            elapsed_time.as_millis(),
            func.function_evaluations()
        );
        func.delete_history();
    }
    println!("{}", res);
}

fn matrix_integration_test_3d() {
    let sim = Simplex3D::new_from_points(
        &array![(8.0f64 / 9.0).sqrt(), 0., -1.0 / 3.0],
        &array![-(2.0f64 / 9.0).sqrt(), (2.0f64 / 3.0).sqrt(), -1.0 / 3.0],
        &array![-(2.0f64 / 9.0).sqrt(), -(2.0f64 / 3.0).sqrt(), -1.0 / 3.0],
        &array![0.0, 0.0, 1.0],
    );

    let mut res = Array2::<f64>::zeros([10, 10]);
    let nodal_values = array![1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let basic_integrator = Quadrilateral3DIntegrator::new(1);
    let hierarchic_inte = Hierarchic3DIntegrator::new(basic_integrator, false, 1e-3);

    let mut cache = Hierarchic3DIntegratorData::new_cache();

    // Zuerst die Hauptdiagonale, dann die Nebendiagonalen
    let res_vec = get_diagonal_order(9);
    //println!("{:?}",res_vec);

    for (count, i, j) in res_vec {
        let func = Box::new(PhaseFieldFuncDiff23D::new(
            nodal_values.clone(),
            1e-6,
            1.,
            i,
            j,
        ));
        let now = Instant::now();
        res[[i, j]] = hierarchic_inte
            .integrate_simplex(&func, &sim, &mut cache)
            .get();
        let elapsed_time = now.elapsed();
        if count < 10 {
            // Wenn Diagonale und erste nebendiagonale durch sind
            cache.make_leafs_unchecked();
        }
        println!(
            "Running [{},{}] took {} milliseconds. Tree size is: {}",
            i,
            j,
            elapsed_time.as_millis(),
            cache.tree_size()
        );
    }
    println!("{}", res);
}

fn all_figures() {
    create_figures(vec![
        (
            String::from("Quad2D-1"),
            Box::new(Quadrilateral2DIntegrator::new(1)),
        ),
        (
            String::from("Quad2D-2"),
            Box::new(Quadrilateral2DIntegrator::new(2)),
        ),
        (
            String::from("Quad2D-3"),
            Box::new(Quadrilateral2DIntegrator::new(3)),
        ),
    ]);
    create_figures(vec![
        (
            String::from("Dunavant2D-1"),
            Box::new(DunavantIntegrator::new(1)),
        ),
        (
            String::from("Dunavant2D-2"),
            Box::new(DunavantIntegrator::new(2)),
        ),
        (
            String::from("Dunavant2D-3"),
            Box::new(DunavantIntegrator::new(3)),
        ),
    ]);
    create_figures(vec![
        (
            String::from("EdgeSubdivision2D-1"),
            Box::new(EdgeSubdivisionIntegrator::new(
                DunavantIntegrator::new(1),
                1,
            )),
        ),
        (
            String::from("EdgeSubdivision2D-2"),
            Box::new(EdgeSubdivisionIntegrator::new(
                DunavantIntegrator::new(1),
                2,
            )),
        ),
        (
            String::from("EdgeSubdivision2D-3"),
            Box::new(EdgeSubdivisionIntegrator::new(
                DunavantIntegrator::new(1),
                3,
            )),
        ),
        (
            String::from("EdgeSubdivision2D-4"),
            Box::new(EdgeSubdivisionIntegrator::new(
                DunavantIntegrator::new(1),
                4,
            )),
        ),
        (
            String::from("EdgeSubdivision2D-5"),
            Box::new(EdgeSubdivisionIntegrator::new(
                DunavantIntegrator::new(1),
                5,
            )),
        ),
        (
            String::from("EdgeSubdivision2D-6"),
            Box::new(EdgeSubdivisionIntegrator::new(
                DunavantIntegrator::new(1),
                6,
            )),
        ),
    ]);
}

fn export_data(str: &str, data: Vec<[f64; 3]>) {
    let mut file = File::create(&format!("{}.csv", str)).unwrap();
    // Header

    //write!(file, "\\begin{{scope}}\n").unwrap();
    for d in data {
        write!(file, "{},{},{}\n", d[0], d[1], d[2]).unwrap();
    }
}

fn edge_subdivision_stats() {
    let data = edge_refinement_test_2d(DunavantIntegrator::new(1));
    export_data(&"Edge-Dunavant-1", data);
    let data = edge_refinement_test_2d(DunavantIntegrator::new(2));
    export_data(&"Edge-Dunavant-2", data);
    let data = edge_refinement_test_2d(DunavantIntegrator::new(3));
    export_data(&"Edge-Dunavant-3", data);

    let data = edge_refinement_test_2d(Quadrilateral2DIntegrator::new(1));
    export_data(&"Edge-Quad-1", data);
    let data = edge_refinement_test_2d(Quadrilateral2DIntegrator::new(2));
    export_data(&"Edge-Quad-2", data);
    let data = edge_refinement_test_2d(Quadrilateral2DIntegrator::new(3));
    export_data(&"Edge-Quad-3", data);
}

fn factorial(arg: usize) -> usize {
    let mut result = 1;
    for i in 0..arg {
        result *= arg - i
    }
    result
}

fn binomial(n: u64, k: u64) -> u64 {
    let mut res = 1;
    for i in 0..k {
        res = (res * (n - i)) / (i + 1);
    }
    res
}

fn num_simplizes(highest_dim: u64, order: u64, dim_simplex: u64) -> i64 {
    let mut result = 0;
    for i in 0..=dim_simplex {
        let c = binomial(highest_dim + (dim_simplex + 1 - i) * order, highest_dim);
        if i % 2 == 0 {
            result += (binomial(dim_simplex, i) * c) as i64
        } else {
            result -= (binomial(dim_simplex, i) * c) as i64
        };
    }
    return result;
}

fn main() {
    //all_figures();

    let sim = Simplex3D::new_from_points(
        &array![(8.0f64 / 9.0).sqrt(), 0., -1.0 / 3.0],
        &array![-(2.0f64 / 9.0).sqrt(), (2.0f64 / 3.0).sqrt(), -1.0 / 3.0],
        &array![-(2.0f64 / 9.0).sqrt(), -(2.0f64 / 3.0).sqrt(), -1.0 / 3.0],
        &array![0.0, 0.0, 1.0],
    );

    let mut res = Array2::<f64>::zeros([10, 10]);
    let nodal_values = array![1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let basic_integrator = Quadrilateral3DIntegrator::new(1);
    let edge_inte = Edge3D::new(basic_integrator, 10);

    let func = Box::new(Constant3DFunction {});

    edge_inte.integrate_simplex(&func, &sim, &mut IntegratorDummy::get());

    println!("{}",sim.get_volume())
    /*
    let sim = Simplex2D::new_from_points(
        &array![(8.0f64 / 9.0).sqrt(), 0., -1.0 / 3.0],
        &array![-(2.0f64 / 9.0).sqrt(), (2.0f64 / 3.0).sqrt(), -1.0 / 3.0],
        &array![-(2.0f64 / 9.0).sqrt(), -(2.0f64 / 3.0).sqrt(), -1.0 / 3.0],
    );

    let integrator = DunavantIntegrator::new(1);
    let hierarchic_integrator = Hierarchic2DIntegrator::new(integrator, false, 1e-7);

    let simplex_func = PhaseFieldFuncMatrix2D::new(array![1.0, 1.0, 1.0, -1.0, 0.0, 0.0], 1e-6, 1.0);
    let func = Box::new(Function2DHistory::new(simplex_func));

    let mut cache = Hierarchic2DIntegratorData::new_cache();

    let result = hierarchic_integrator.integrate_simplex(&func, &sim, &mut cache);

    func.delete_history();

    let result = hierarchic_integrator.integrate_simplex(&func, &sim, &mut cache);

    let hist = func.get_history();

    println!("{},{}", hist.len(), result.get_borrow());
    let mut file = File::create(&"out.csv").unwrap();
    for el in &hist {
        //println!("{},{}",el, el.fold(0., |f1, f2| f1 + f2));
        let sim_points = sim.get_points();
        let el = el;
        let point = sim_points.dot(el);
        write!(file, "{} {}\n", point[0], point[1]).unwrap();
        //println!(
        //    "\\draw[fill,red] (barycentric cs:b1={:.3},b2={:.3},b3={:.3},b4={:.3}) circle (2pt);",
        //    el[0], el[1], el[2], el[3]
        //);
    }
    */
    //println!("{},{}", result.get(), sim.get_area());

    //matrix_integration_test_2d()

    /*
    let order = 4;
    for i in 0..=order {
        for j in 0..=(order - i) {
            for k in 0..=(order - i - j) {
                let m = order - i - j - k;
                println!("{},{},{},{}", j, i, k, m);
                let (i0, i1, i2, i3) = (j, i, k, m);
            }
        }
    }

    let dim = 2;
    for i in 0..10 {
        //println!("{}! = {}",i,factorial(i));
        println!(
            "3-Simplex in {}th Subdivision: {}",
            i,
            num_simplizes(dim, i, 2)
        );
        println!(
            "1-Simplex in {}th Subdivision: {}",
            i,
            num_simplizes(dim, i, 0)
        )
    }*/
}
