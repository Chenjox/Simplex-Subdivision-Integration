
mod shape_func {
    /// Shape Function for the 1 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN1(xi1: f64, _xi2: f64, _xi3: f64) -> f64 {
        2.0*xi1*(xi1 - 0.5)
    }
    /// Shape Function for the 2 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN2(_xi1: f64, xi2: f64, _xi3: f64) -> f64 {
        2.0*xi2*(xi2 - 0.5)
    }
    /// Shape Function for the 3 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN3(_xi1: f64, _xi2: f64, xi3: f64) -> f64 {
        2.0*xi3*(xi3 - 0.5)
    }
    /// Shape Function for the 4 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN4(xi1: f64, _xi2: f64, xi3: f64) -> f64 {
        -2.0*xi1*(xi1 - 1.0)-2.0*xi3*(xi3 - 1.0)
    }
    /// Shape Function for the 5 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN5(xi1: f64, xi2: f64, _xi3: f64) -> f64 {
        -2.0*xi1*(xi1 - 1.0)-2.0*xi2*(xi2 - 1.0)
    }
    /// Shape Function for the 6 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN6(_xi1: f64, xi2: f64, xi3: f64) -> f64 {
        -2.0*xi2*(xi2 - 1.0)-2.0*xi3*(xi3 - 1.0)
    }
}