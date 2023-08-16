
use ndarray::Array2;

use crate::integration_2d::domain::{
  IntegratorDummy, Simplex2D, Simplex2DFunction, Simplex2DIntegrator,
};

pub struct EdgeSubdivisionIntegrator<I: Simplex2DIntegrator<IntegratorDummy>> {
  base_integrator: I,
  order: usize
}

impl<I: Simplex2DIntegrator<IntegratorDummy>> Simplex2DIntegrator<IntegratorDummy>
    for EdgeSubdivisionIntegrator<I>
{
    fn integrate_over_domain<T: Simplex2DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex2D,
        cached_data: &mut IntegratorDummy,
    ) -> f64 {
      todo!("Implement counting scheme");
    }
}