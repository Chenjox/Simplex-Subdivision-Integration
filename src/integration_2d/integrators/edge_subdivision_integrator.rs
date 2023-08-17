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
                    [i0 as f64 / order_fl,     i1 as f64/order_fl, i2 as f64 / order_fl],
                    [(i0-1) as f64 / order_fl, i1 as f64/order_fl, (i2+1) as f64 / order_fl],
                    [i0 as f64 / order_fl,     (i1-1) as f64/order_fl, (i2+1) as f64 / order_fl],
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
                  [i0 as f64 / order_fl,     i1 as f64/order_fl, i2 as f64 / order_fl],
                  [(i0-1) as f64 / order_fl, (i1+1) as f64/order_fl, (i2) as f64 / order_fl],
                  [(i0-1) as f64 / order_fl, i1 as f64/order_fl,     (i2+1) as f64 / order_fl],
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

#[cfg(test)]
mod tests {
    use crate::{integrator_tests, integration_2d::domain::IntegratorDummy};
    use crate::integration_2d::integrators::{EdgeSubdivisionIntegrator, Quadrilateral2DIntegrator};

    integrator_tests!{
        order2: EdgeSubdivisionIntegrator<Quadrilateral2DIntegrator>: EdgeSubdivisionIntegrator::new(Quadrilateral2DIntegrator::new(1),2), IntegratorDummy: IntegratorDummy::get(),
        order3: EdgeSubdivisionIntegrator<Quadrilateral2DIntegrator>: EdgeSubdivisionIntegrator::new(Quadrilateral2DIntegrator::new(1),3), IntegratorDummy: IntegratorDummy::get(),
        order4: EdgeSubdivisionIntegrator<Quadrilateral2DIntegrator>: EdgeSubdivisionIntegrator::new(Quadrilateral2DIntegrator::new(1),4), IntegratorDummy: IntegratorDummy::get(),
    }
}