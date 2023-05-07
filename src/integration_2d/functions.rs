use crate::integration_2d::domain::*;
use ndarray::{array, Array1};
use std::cell::RefCell;

/// A Dummy Struct implementing a Constant function for the given Simplex.
pub struct Constant2DFunction;

/// Implementation of the trait [`Simplex2DFunction`] for the struct [`Constant2DFunction`]
impl Simplex2DFunction for Constant2DFunction {
    fn function(&self, _xi1: f64, _xi2: f64, _xi3: f64, _simplex: &Simplex2D) -> f64 {
        1.0
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
    history: RefCell<Vec<(Array1<f64>, f64)>>,
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
    pub fn get_history(self) -> Vec<(Array1<f64>, f64)> {
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
    fn function(&self, xi1: f64, xi2: f64, xi3: f64, simplex: &Simplex2D) -> f64 {
        let result = self.function.function(xi1, xi2, xi3, simplex);
        {
            let mut history = self.history.borrow_mut();
            history.push((array![xi1, xi2, xi3], result));
        }
        result
    }
}
