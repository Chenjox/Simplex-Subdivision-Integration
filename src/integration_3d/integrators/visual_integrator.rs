//!
//! These integrators don't integrate, but they instead evaluate at specific points in order 
//! for figures to be produced.

use ndarray::{Array2, Array1, array};

use crate::integration_3d::{domain::{Simplex3D, Simplex3DFunction, Simplex3DIntegrator}, IntegratorDummy};

pub struct OrientationChecker;

impl OrientationChecker {
  fn point_order(num: usize) -> Array1<f64> {
    match num {
      0 => array![ 1.0, 0.0 ,0.0 ,0.0],
      1 => array![ 0.0, 1.0, 0.0, 0.0],
      2 => array![ 0.0, 0.0, 1.0, 0.0],
      3 => array![ 0.0, 0.0, 0.0, 1.0],
      _ => Self::point_order(0)
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
    ) -> f64 {
      let mut result = 0.;
      for i in 0..4 {
        let point = Self::point_order(i);
        let point = transformation.dot(&point);

        result += func.function_vec(&point, simplex);
      }
      return result;
    }
}