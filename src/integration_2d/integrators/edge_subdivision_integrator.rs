use ndarray::{Array2, array};

use crate::integration_2d::domain::{
    IntegratorDummy, Simplex2D, Simplex2DFunction, Simplex2DIntegrator,
};

pub struct EdgeSubdivisionIntegrator<I: Simplex2DIntegrator<IntegratorDummy>> {
    base_integrator: I,
    order: usize,
}

impl<I: Simplex2DIntegrator<IntegratorDummy>> EdgeSubdivisionIntegrator<I> {
    pub fn new(base_integrator: I, order: usize) -> Self {
      return EdgeSubdivisionIntegrator {
        base_integrator,
        order
      };
    }
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
        let order = self.order;
        let order_fl = order as f64;

        let mut result = 0.;

        let order = order - 1;
        for i in 0..=order {
            for j in 0..=(order - i) {
                let k = order - i - j;
                //println!("{},{},{}", j, i, k);
                let (i0,i1,i2) = (j,i,k);
                // 0,-1,1
                let i0 = i0 +1;
                if i1 != 0 {
                  let transformation = array![
                    [i0 as f64 / order as f64,     (i1-1) as f64/order as f64, (i2+1) as f64 / order as f64],
                    [i0 as f64 / order as f64,     i1 as f64/order as f64, i2 as f64 / order as f64],
                    [(i0-1) as f64 / order as f64, i1 as f64/order as f64, (i2+1) as f64 / order as f64]
                  ];
                  let transformation = transformation.reversed_axes();
                  result += self.base_integrator.integrate_over_domain(&transformation, func, simplex, &mut IntegratorDummy);
                  //println!("{},{},{}", i0,i1,i2);
                  //println!("{},{},{}", i0,i1-1,i2+1);
                  ////
                  //println!("{},{},{}", i0-1,i1,i2+1);
                }
                // -1,1,0
                let transformation = array![
                  [i0 as f64 / order as f64,     i1 as f64/order as f64, i2 as f64 / order as f64],
                    [(i0-1) as f64 / order as f64,     (i1+1) as f64/order as f64, (i2) as f64 / order as f64],
                    [(i0-1) as f64 / order as f64, i1 as f64/order as f64, (i2+1) as f64 / order as f64]
                ];
                let transformation = transformation.reversed_axes();
                result += self.base_integrator.integrate_over_domain(&transformation, func, simplex, &mut IntegratorDummy);
                //println!("{},{},{}", i0,i1,i2);
                //println!("{},{},{}", i0-1,i1+1,i2+0);

                //println!("{},{},{}", i0-1,i1,i2+1);
                
            }
        }
        return result;
    }
}
