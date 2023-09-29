use crate::integration_3d::domain::*;
use ndarray::{array, Array1};
use std::cell::RefCell;

pub struct Constant3DFunction;

impl Simplex3DFunction for Constant3DFunction {
    type Return = ResultTypeWrapper<f64>;
    fn function(&self, _xi1: f64, _xi2: f64, _xi3: f64, _xi4: f64, _simplex: &Simplex3D) -> Self::Return {
        ResultTypeWrapper::new(1.0)
    }
}

pub struct Multiplicative3DFunction;

impl Simplex3DFunction for Multiplicative3DFunction {
    type Return = ResultTypeWrapper<f64>;
    fn function(&self, xi1: f64, xi2: f64, xi3: f64, xi4: f64, _simplex: &Simplex3D) -> Self::Return {
        return ResultTypeWrapper::new(((0.5 - 4. * xi1 * xi2 * xi3 * xi4).abs()).exp());
    }
}

/// A struct which will record all function evaluations of the given [`Simplex3DFunction`]
pub struct Function3DHistory<F: Simplex3DFunction> {
    history: RefCell<Vec<Array1<f64>>>,
    function: F,
}

impl<F: Simplex3DFunction> Function3DHistory<F> {
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

impl<F: Simplex3DFunction> Simplex3DFunction for Function3DHistory<F> {
    type Return = F::Return;

    fn function(&self, xi1: f64, xi2: f64, xi3: f64, xi4: f64, simplex: &Simplex3D) -> Self::Return {
        let result = self.function.function(xi1, xi2, xi3, xi4, simplex);
        {
            let mut history = self.history.borrow_mut();
            history.push(array![xi1, xi2, xi3, xi4]);
        }
        result
    }
}
