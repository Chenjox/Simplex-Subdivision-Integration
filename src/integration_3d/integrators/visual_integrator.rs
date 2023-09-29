//!
//! These integrators don't integrate, but they instead evaluate at specific points in order
//! for figures to be produced.

use std::ops::AddAssign;

use ndarray::{array, Array1, Array2};

use crate::common::IntegratorDummy;
use crate::integration_3d::{
    domain::{Simplex3D, Simplex3DFunction, Simplex3DIntegrator},
    Simplex3DResultType,
};

pub struct OrientationChecker;

impl OrientationChecker {
    fn point_order(num: usize) -> Array1<f64> {
        match num {
            0 => array![1.0, 0.0, 0.0, 0.0],
            1 => array![0.0, 1.0, 0.0, 0.0],
            2 => array![0.0, 0.0, 1.0, 0.0],
            3 => array![0.0, 0.0, 0.0, 1.0],
            _ => Self::point_order(0),
        }
    }
}

impl Simplex3DIntegrator<IntegratorDummy> for OrientationChecker {
    fn integrate_over_domain<T: Simplex3DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex3D,
        _cache_data: &mut IntegratorDummy,
    ) -> T::Return {
        let mut result = func.additive_neutral_element();
        for i in 0..4 {
            let point = Self::point_order(i);
            let point = transformation.dot(&point);

            Simplex3DResultType::add_assign(&mut result, &func.function_vec(&point, simplex));
        }
        return result;
    }
}
