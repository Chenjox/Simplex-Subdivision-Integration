use ndarray::{array, Array2};
use std::fs::File;
use std::io::Write;

use crate::{
    integration_2d::{
        domain::{IntegratorDummy, Simplex2D, Simplex2DIntegrator},
        functions::{Function2DHistory, RepeatedPyramidFunction},
        integrators::EdgeSubdivisionIntegrator,
    },
    problems::problem_definition::problem_2d_definition::PhaseFieldFuncDiff22D,
};

/// Creates the Figures for documentation
pub fn create_figures(integrators: Vec<(String, Box<impl Simplex2DIntegrator<IntegratorDummy>>)>) {
    let sim = Simplex2D::new_from_points(
        &array![0., 0.],
        &array![1., 0.],
        &array![0.5, (3.0f64).sqrt() / 2.],
    );
    let mut dummy = IntegratorDummy::get();

    let simpoints = sim.get_points();

    for (name, inte) in integrators {
        let func = Box::new(Function2DHistory::new(RepeatedPyramidFunction::new(vec![
            [0.2, 0.3, 0.5, 3.0],
            [0.5, 0.3, 0.2, 3.0],
        ])));

        let result = inte.integrate_simplex(&func, &sim, &mut dummy);

        let hist = func.get_history();
        let mut file = File::create(&format!("{}.tikz", name)).unwrap();
        // Header

        //write!(file, "\\begin{{scope}}\n").unwrap();
        for i in 0..3 {
            let p = simpoints.column(i);
            write!(
                file,
                "\\coordinate (b{}) at ({:.3},{:.3});\n",
                i + 1,
                p[0],
                p[1]
            )
            .unwrap();
        }
        write!(file, "\\draw (b1) -- (b2) -- (b3) --cycle;\n").unwrap();
        for el in &hist {
            //println!("{},{}",el, el.fold(0., |f1, f2| f1 + f2));
            let el = &el.0;
            write!(
                file,
                "\\draw[fill,red] (barycentric cs:b1={:.3},b2={:.3},b3={:.3}) circle (0.5mm);\n",
                el[0], el[1], el[2]
            )
            .unwrap();
        }
        //write!(file, "\\end{{scope}}\n").unwrap();
    }
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

pub fn froebenius_norm(first: Array2<f64>) -> f64 {
    let mut sum = 0.;
    for l in first.into_iter() {
        sum += l.powi(2);
    }
    return sum;
}

pub fn edge_refinement_test_2d<I: Simplex2DIntegrator<IntegratorDummy>>(base_integrator: I) {
    let sim = Simplex2D::new_from_points(
        &array![0., 0.],
        &array![1., 0.],
        &array![0.5, (3.0f64).sqrt() / 2.],
    );
    let mut dummy = IntegratorDummy::get();

    let mut order = 1;

    let nodal_values = array![1.0, 1.0, 1.0, -1.0, 0.0, 0.0];
    let mut evals = 0;
    let mut last_res = Array2::<f64>::zeros([6, 6]);
    let mut res = Array2::<f64>::zeros([6, 6]);
    let res_vec = get_diagonal_order(5);
    loop {
        let edge_integrator = EdgeSubdivisionIntegrator::new(base_integrator.dupe(), order);
        last_res = res.clone();
        for (_count, i, j) in &res_vec {
            let func = Box::new(Function2DHistory::new(PhaseFieldFuncDiff22D::new(
                nodal_values.clone(),
                1e-6,
                1.,
                *i,
                *j,
            )));
            res[[*i, *j]] = edge_integrator.integrate_simplex(&func, &sim, &mut dummy); //&mut cache

            evals += func.function_evaluations();
        }

        if order > 3 {
            let diff = &last_res - &res;
            let norm = froebenius_norm(diff);
            if norm < 1e-5 {
                break;
            }
        }

        println!("{}: {}, {}", order, res, evals);
        if order > 2 {
            break;
        }
        order += 1;
    }
}
