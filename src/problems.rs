pub mod shape_func {
    /// Shape Function for the 1 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN1(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if xi1 + xi2 + xi3 > 1.0 {
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
        if xi1 + xi2 + xi3 > 1.0 {
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
        if xi1 + xi2 + xi3 > 1.0 {
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
        if xi1 + xi2 + xi3 > 1.0 {
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
        if xi1 + xi2 + xi3 > 1.0 {
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
        if xi1 + xi2 + xi3 > 1.0 {
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
