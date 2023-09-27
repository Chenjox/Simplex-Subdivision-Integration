use ndarray::{array, Array2};

use crate::integration_3d::domain::{
    det3x3, IntegratorDummy, Simplex3D, Simplex3DFunction, Simplex3DIntegrator, Simplex3DResultType,
};

pub struct EdgeSubdivisionIntegrator<I: Simplex2DIntegrator<IntegratorDummy>> {
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
    fn dupe(&self) -> Self {
        return Self {
            base_integrator: self.base_integrator.dupe(),
            order: self.order,
        };
    }

    fn integrate_over_domain<T: Simplex2DFunction>(
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
                let k = order - i - j;
                //println!("{},{},{}", j, i, k);
                let (i0, i1, i2) = (j, i, k);
                // 0,-1,1
                let i0 = i0 + 1;
                if i1 != 0 {
                    let ch_transformation = array![
                        [
                            i0 as f64 / order_fl,
                            i1 as f64 / order_fl,
                            i2 as f64 / order_fl
                        ],
                        [
                            (i0 - 1) as f64 / order_fl,
                            i1 as f64 / order_fl,
                            (i2 + 1) as f64 / order_fl
                        ],
                        [
                            i0 as f64 / order_fl,
                            (i1 - 1) as f64 / order_fl,
                            (i2 + 1) as f64 / order_fl
                        ],
                    ];
                    let ch_transformation = ch_transformation.reversed_axes();
                    let transformation = transformation.dot(&ch_transformation);

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
                    //println!("{},{},{}", i0,i1-1,i2+1);
                    ////
                    //println!("{},{},{}", i0-1,i1,i2+1);
                }
                // -1,1,0
                let ch_transformation = array![
                    [
                        i0 as f64 / order_fl,
                        i1 as f64 / order_fl,
                        i2 as f64 / order_fl
                    ],
                    [
                        (i0 - 1) as f64 / order_fl,
                        (i1 + 1) as f64 / order_fl,
                        (i2) as f64 / order_fl
                    ],
                    [
                        (i0 - 1) as f64 / order_fl,
                        i1 as f64 / order_fl,
                        (i2 + 1) as f64 / order_fl
                    ],
                ];
                let ch_transformation = ch_transformation.reversed_axes();
                let transformation = transformation.dot(&ch_transformation);
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
        return result;
    }
}

