use crate::integration_2d::domain::*;
use ndarray::{array, Array1};
use std::cell::RefCell;

const TOLERANCE: f64 = 1e-10;

pub struct PyramidFunction {
    xi1p: f64,
    xi2p: f64,
    xi3p: f64,
    height: f64,
}

impl PyramidFunction {
    pub fn new(xi1p: f64, xi2p: f64, xi3p: f64, height: f64) -> Self {
        if (xi1p + xi2p + xi3p - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1p,
                xi2p,
                xi3p,
                xi1p + xi2p + xi3p
            );
        };
        PyramidFunction {
            xi1p,
            xi2p,
            xi3p,
            height,
        }
    }
}

impl Simplex2DFunction for PyramidFunction {
    type Return = ResultTypeWrapper<f64>;
    fn function(&self, xi1: f64, xi2: f64, xi3: f64, simplex: &Simplex2D) -> ResultTypeWrapper<f64> {
        ResultTypeWrapper::new(self.height * (xi1 / self.xi1p).min(xi2 / self.xi2p).min(xi3 / self.xi3p))
    }
}

pub struct RepeatedPyramidFunction {
    pyramids: Vec<[f64; 4]>,
}
impl RepeatedPyramidFunction {
    pub fn new(pyramids: Vec<[f64; 4]>) -> Self {
        /*if (xi1p + xi2p + xi3p - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1p,
                xi2p,
                xi3p,
                xi1p + xi2p + xi3p
            );
        };*/
        Self { pyramids }
    }
    fn get_pyramid_value(xi1: f64, xi2: f64, xi3: f64, a: &[f64; 4]) -> f64 {
        a[3] * (xi1 / a[0]).min(xi2 / a[1]).min(xi3 / a[2])
    }
}
impl Simplex2DFunction for RepeatedPyramidFunction {
    type Return = ResultTypeWrapper<f64>;
    fn function(&self, xi1: f64, xi2: f64, xi3: f64, simplex: &Simplex2D) -> Self::Return {
        let mut sum = 0.0;
        for i in 0..self.pyramids.len() {
            let a = &self.pyramids[i];
            sum += Self::get_pyramid_value(xi1, xi2, xi3, &a)
        }
        return ResultTypeWrapper::new(sum);
    }
}

/// A Dummy Struct implementing a Constant function for the given Simplex.
pub struct Constant2DFunction;

/// Implementation of the trait [`Simplex2DFunction`] for the struct [`Constant2DFunction`]
impl Simplex2DFunction for Constant2DFunction {
    type Return = ResultTypeWrapper<f64>;
    fn function(&self, _xi1: f64, _xi2: f64, _xi3: f64, _simplex: &Simplex2D) -> Self::Return {
        ResultTypeWrapper::new(1.0)
    }
}

pub type Constant2DFunctionHistory = Function2DHistory<Constant2DFunction>;

impl Constant2DFunctionHistory {
    /// Helpful constructor function for [`Constant2DFunction`]
    pub fn new_constant() -> Self {
        return Function2DHistory::new(Constant2DFunction {});
    }
}

/// A struct which will record all function evaluations of the given [`Simplex2DFunction`]
pub struct Function2DHistory<F: Simplex2DFunction> {
    history: RefCell<Vec<Array1<f64>>>,
    function: F,
}

impl<F: Simplex2DFunction> Function2DHistory<F> {
    pub fn new(func: F) -> Self {
        return Self {
            history: RefCell::new(Vec::new()),
            function: func,
        };
    }
    /// A history of all function evaluations will be returned.
    /// Drops the struct upon after this function.
    pub fn get_history(self) -> Vec<Array1<f64>> {
        return self.history.take();
    }

    pub fn function_evaluations(&self) -> usize {
        return self.history.borrow().len();
    }

    /// Deletes the history of the function evaluations.
    pub fn delete_history(&self) {
        self.history.borrow_mut().clear();
    }
}

impl<F: Simplex2DFunction> Simplex2DFunction for Function2DHistory<F> {
    type Return = F::Return;
    fn function(&self, xi1: f64, xi2: f64, xi3: f64, simplex: &Simplex2D) -> F::Return {
        let result = self.function.function(xi1, xi2, xi3, simplex);
        {
            let mut history = self.history.borrow_mut();
            history.push(array![xi1, xi2, xi3]);
        }
        result
    }
}
