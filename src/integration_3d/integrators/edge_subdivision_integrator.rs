use ndarray::{array, Array2};

use crate::common::{IntegratorDummy, det3x3};
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
        real_transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex3D,
        _cached_data: &mut IntegratorDummy,
    ) -> T::Return {
        let order = self.order;
        let order_fl = order as f64;

        let mut result = func.additive_neutral_element();
        //todo!("Generalize to 3 Dimensions, currently it's only two.");
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
                    //println!("{:2.0},{:2.0},{:2.0},{:2.0}", i0, i1, i2, i3);

                    // Dieser Fall geht immer
                    let ch_transformation = array![
                        [i0 / order_fl, i1 / order_fl, i2 / order_fl, i3 / order_fl],
                        [
                            (i0 - 1.) / order_fl,
                            (i1 + 1.) / order_fl,
                            i2 / order_fl,
                            i3 / order_fl
                        ],
                        [
                            (i0 - 1.) / order_fl,
                            i1 / order_fl,
                            (i2 + 1.) / order_fl,
                            i3 / order_fl
                        ],
                        [
                            (i0 - 1.) / order_fl,
                            i1 / order_fl,
                            i2 / order_fl,
                            (i3 + 1.) / order_fl
                        ]
                    ];
                    let ch_transformation = ch_transformation.reversed_axes();
                    let transformation = real_transformation.dot(&ch_transformation);

                    //let ch_transformation = ch_transformation.reversed_axes();
                    //let transformation = transformation.dot(&ch_transformation);
                    result.add_assign(&{
                        let r = self.base_integrator.integrate_over_domain(
                            &transformation,
                            func,
                            simplex,
                            &mut IntegratorDummy,
                        );
                        //println!("Tet 1: {}", det3x3(&transformation));
                        r
                    });

                    // Oktaederfall
                    if i0 >= 2. {
                        let ch_transformation = array![
                            [
                                (i0 - 1.) / order_fl,
                                i1 / order_fl,
                                i2 / order_fl,
                                (i3 + 1.) / order_fl
                            ],
                            [
                                (i0 - 1.) / order_fl,
                                (i1 + 1.) / order_fl,
                                i2 / order_fl,
                                i3 / order_fl
                            ],
                            [
                                (i0 - 1.) / order_fl,
                                i1 / order_fl,
                                (i2 + 1.) / order_fl,
                                i3 / order_fl
                            ],
                            [
                                (i0 - 2.) / order_fl,
                                (i1 + 1.) / order_fl,
                                (i2 + 1.) / order_fl,
                                i3 / order_fl
                            ]
                        ];
                        let ch_transformation = ch_transformation.reversed_axes();
                        //println!("DetOkt1 = {}", det4x4(&ch_transformation));
                        let transformation = real_transformation.dot(&ch_transformation);

                        //let ch_transformation = ch_transformation.reversed_axes();
                        //let transformation = transformation.dot(&ch_transformation);
                        result.add_assign(&{
                            let r = self.base_integrator.integrate_over_domain(
                                &transformation,
                                func,
                                simplex,
                                &mut IntegratorDummy,
                            );
                            println!("Okt 1: {}", det3x3(&transformation));
                            r
                        });

                        let ch_transformation = array![
                            [
                                (i0 - 1.) / order_fl,
                                i1 / order_fl,
                                (i2 + 1.) / order_fl,
                                i3 / order_fl
                            ],
                            [
                                (i0 - 2.) / order_fl,
                                i1 / order_fl,
                                (i2 + 1.) / order_fl,
                                (i3 + 1.) / order_fl
                            ],
                            [
                                (i0 - 1.) / order_fl,
                                i1 / order_fl,
                                i2 / order_fl,
                                (i3 + 1.) / order_fl
                            ],
                            [
                                (i0 - 2.) / order_fl,
                                (i1 + 1.) / order_fl,
                                (i2 + 1.) / order_fl,
                                i3 / order_fl
                            ]
                        ];
                        let ch_transformation = ch_transformation.reversed_axes();
                        //println!("DetOkt2 = {}", det4x4(&ch_transformation));
                        let transformation = real_transformation.dot(&ch_transformation);

                        //let ch_transformation = ch_transformation.reversed_axes();
                        //let transformation = transformation.dot(&ch_transformation);
                        result.add_assign(&{
                            let r = self.base_integrator.integrate_over_domain(
                                &transformation,
                                func,
                                simplex,
                                &mut IntegratorDummy,
                            );
                            //println!("Okt 2: {}", det3x3(&transformation));
                            r
                        });

                        let ch_transformation = array![
                            [
                                (i0 - 2.) / order_fl,
                                (i1 + 1.) / order_fl,
                                (i2 + 1.) / order_fl,
                                i3 / order_fl
                            ],
                            [
                                (i0 - 2.) / order_fl,
                                i1 / order_fl,
                                (i2 + 1.) / order_fl,
                                (i3 + 1.) / order_fl
                            ],
                            [
                                (i0 - 1.) / order_fl,
                                i1 / order_fl,
                                i2 / order_fl,
                                (i3 + 1.) / order_fl
                            ],
                            [
                                (i0 - 2.) / order_fl,
                                (i1 + 1.) / order_fl,
                                i2 / order_fl,
                                (i3 + 1.) / order_fl
                            ]
                        ];
                        let ch_transformation = ch_transformation.reversed_axes();
                        //println!("DetOkt3 = {}", det4x4(&ch_transformation));
                        let transformation = real_transformation.dot(&ch_transformation);

                        //let ch_transformation = ch_transformation.reversed_axes();
                        //let transformation = transformation.dot(&ch_transformation);
                        result.add_assign(&{
                            let r = self.base_integrator.integrate_over_domain(
                                &transformation,
                                func,
                                simplex,
                                &mut IntegratorDummy,
                            );
                            //println!("Okt 3: {}", det3x3(&transformation));
                            r
                        });

                        let ch_transformation = array![
                            [
                                (i0 - 2.) / order_fl,
                                (i1 + 1.) / order_fl,
                                i2 / order_fl,
                                (i3 + 1.) / order_fl
                            ],
                            [
                                (i0 - 2.) / order_fl,
                                (i1 + 1.) / order_fl,
                                (i2 + 1.) / order_fl,
                                i3 / order_fl
                            ],
                            [
                                (i0 - 1.) / order_fl,
                                (i1 + 1.) / order_fl,
                                i2 / order_fl,
                                i3 / order_fl
                            ],
                            [
                                (i0 - 1.) / order_fl,
                                i1 / order_fl,
                                i2 / order_fl,
                                (i3 + 1.) / order_fl
                            ]
                        ];
                        let ch_transformation = ch_transformation.reversed_axes();
                        //println!("DetOkt4 = {}", det4x4(&ch_transformation));
                        let transformation = real_transformation.dot(&ch_transformation);

                        //let ch_transformation = ch_transformation.reversed_axes();
                        //let transformation = transformation.dot(&ch_transformation);
                        result.add_assign(&{
                            let r = self.base_integrator.integrate_over_domain(
                                &transformation,
                                func,
                                simplex,
                                &mut IntegratorDummy,
                            );
                            //println!("Okt 4: {}", det3x3(&transformation));
                            r
                        });
                    }
                    // Umgedrehter Tetraederfall
                    if i0 >= 3. {
                        let ch_transformation = array![
                            [
                                (i0 - 2.) / order_fl,
                                i1 / order_fl,
                                (i2 + 1.) / order_fl,
                                (i3 + 1.) / order_fl
                            ],
                            [
                                (i0 - 2.) / order_fl,
                                (i1 + 1.) / order_fl,
                                i2 / order_fl,
                                (i3 + 1.) / order_fl
                            ],
                            [
                                (i0 - 2.) / order_fl,
                                (i1 + 1.) / order_fl,
                                (i2 + 1.) / order_fl,
                                i3 / order_fl
                            ],
                            [
                                (i0 - 3.) / order_fl,
                                (i1 + 1.) / order_fl,
                                (i2 + 1.) / order_fl,
                                (i3 + 1.) / order_fl
                            ]
                        ];
                        let ch_transformation = ch_transformation.reversed_axes();
                        //println!("DetTet2 = {}", det4x4(&ch_transformation));
                        let transformation = real_transformation.dot(&ch_transformation);

                        //let ch_transformation = ch_transformation.reversed_axes();
                        //let transformation = transformation.dot(&ch_transformation);
                        result.add_assign(&{
                            let r = self.base_integrator.integrate_over_domain(
                                &transformation,
                                func,
                                simplex,
                                &mut IntegratorDummy,
                            );
                            println!("Tet 2: {}", det3x3(&transformation));
                            r
                        });
                    }
                }
            }
        }
        return result;
    }
}


#[cfg(test)]
mod tests {
    use crate::common::IntegratorDummy;
    use crate::integration_3d::integrators::{
        EdgeSubdivisionIntegrator, Quadrilateral3DIntegrator,
    };
    use crate::integrator_tests_3d;

    integrator_tests_3d! {
        order2_quad: EdgeSubdivisionIntegrator<Quadrilateral3DIntegrator>: EdgeSubdivisionIntegrator::new(Quadrilateral3DIntegrator::new(2),2), IntegratorDummy: IntegratorDummy::get(),
        order3_quad: EdgeSubdivisionIntegrator<Quadrilateral3DIntegrator>: EdgeSubdivisionIntegrator::new(Quadrilateral3DIntegrator::new(2),3), IntegratorDummy: IntegratorDummy::get(),
        order4_quad: EdgeSubdivisionIntegrator<Quadrilateral3DIntegrator>: EdgeSubdivisionIntegrator::new(Quadrilateral3DIntegrator::new(2),4), IntegratorDummy: IntegratorDummy::get(),
    }
}