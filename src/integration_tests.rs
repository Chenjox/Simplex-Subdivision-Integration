use ndarray::array;
use std::fs::File;
use std::io::Write;

use crate::integration_2d::{
    domain::{IntegratorDummy, Simplex2D, Simplex2DIntegrator},
    functions::{Function2DHistory, RepeatedPyramidFunction},
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
