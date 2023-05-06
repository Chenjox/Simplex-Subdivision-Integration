use crate::domain::Simplex2DFunction;

use self::{phase_field::phase_field_func, shape_func::approx_func};

const TOLERANCE: f64 = 1e-10;

pub mod shape_func {
    use super::TOLERANCE;

    /// Shape Function for the 1 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN1(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        2.0 * xi1 * (xi1 - 0.5)
    }
    /// Shape Function for the 2 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN2(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        2.0 * xi2 * (xi2 - 0.5)
    }
    /// Shape Function for the 3 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN3(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        2.0 * xi3 * (xi3 - 0.5)
    }
    /// Shape Function for the 4 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN4(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        4.0 * xi1 * xi3
    }
    /// Shape Function for the 5 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN5(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        4.0 * xi1 * xi2
    }
    /// Shape Function for the 6 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN6(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        4.0 * xi2 * xi3
    }

    pub fn approx_func(weights: [f64; 6], xi1: f64, xi2: f64, xi3: f64) -> f64 {
        weights[0] * BarN1(xi1, xi2, xi3)
            + weights[1] * BarN2(xi1, xi2, xi3)
            + weights[2] * BarN3(xi1, xi2, xi3)
            + weights[3] * BarN4(xi1, xi2, xi3)
            + weights[4] * BarN5(xi1, xi2, xi3)
            + weights[5] * BarN6(xi1, xi2, xi3)
    }
}

pub mod phase_field {
    pub fn phase_field_func(fbase: f64, kreg: f64, l: f64) -> f64 {
        (-fbase / (fbase.powi(2) + kreg).powf(0.25) * 1.0 / l).exp()
    }
}

pub struct PhaseField2DFunction {
    pub weights: [f64; 6],
}

impl Simplex2DFunction for PhaseField2DFunction {
    fn function(&self, xi1: f64, xi2: f64, xi3: f64, _simplex: &crate::domain::Simplex2D) -> f64 {
        let f_base = approx_func(self.weights, xi1, xi2, xi3);
        return phase_field_func(f_base, 1e-6, 1.0);
    }
}
