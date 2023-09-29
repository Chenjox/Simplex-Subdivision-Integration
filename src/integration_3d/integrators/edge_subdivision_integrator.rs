use ndarray::{array, Array2};

use crate::common::IntegratorDummy;
use crate::integration_3d::domain::{
    det4x4, Simplex3D, Simplex3DFunction, Simplex3DIntegrator, Simplex3DResultType,
};

pub struct EdgeSubdivisionIntegrator<I: Simplex3DIntegrator<IntegratorDummy>> {
    base_integrator: I,
    order: usize,
}

impl<I: Simplex3DIntegrator<IntegratorDummy>> EdgeSubdivisionIntegrator<I> {
    pub fn new(base_integrator: I, order: usize) -> Self {
        return EdgeSubdivisionIntegrator {
            base_integrator,
            order,
        };
    }
}

impl<I: Simplex3DIntegrator<IntegratorDummy>> Simplex3DIntegrator<IntegratorDummy>
    for EdgeSubdivisionIntegrator<I>
{
    fn integrate_over_domain<T: Simplex3DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex3D,
        cached_data: &mut IntegratorDummy,
    ) -> T::Return {
        let order = self.order;
        let order_fl = order as f64;

        let mut result = func.additive_neutral_element();
        todo!("Generalize to 3 Dimensions, currently it's only two.");
        let order = order - 1;
        for i in 0..=order {
            for j in 0..=(order - i) {
                for k in 0..=(order - i - j) {
                    // Die vierte Koordinate
                    let m = order - i - j - k;
                    //println!("{},{},{},{}", j, i, k, m);
                    // Umbenennung
                    let (i0, i1, i2, i3) = (j as f64, i as f64, k as f64, m as f64);
                    // Shift zum größeren Simplex
                    let i0 = i0 + 1.;
                    //
                    println!("{:2.2},{:2.2},{:2.2},{:2.2}", i0, i1, i2, i3);

                    //let ch_transformation = ch_transformation.reversed_axes();
                    //let transformation = transformation.dot(&ch_transformation);
                    result.add_assign(&{
                        let r = self.base_integrator.integrate_over_domain(
                            &transformation,
                            func,
                            simplex,
                            &mut IntegratorDummy,
                        );
                        //println!("0: {}", det3x3(&transformation));
                        r
                    })
                    //println!("{},{},{}", i0,i1,i2);
                    //println!("{},{},{}", i0-1,i1+1,i2+0);

                    //println!("{},{},{}", i0-1,i1,i2+1);
                }
            }
        }
        return result;
    }
}
