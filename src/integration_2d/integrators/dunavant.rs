use ndarray::{array, Array1, Array2};

use crate::integration_2d::domain::{
    IntegratorDummy, Simplex2D, Simplex2DFunction, Simplex2DIntegrator,
};

fn det2x2(mat2x2: &Array2<f64>) -> f64 {
    mat2x2[[0, 0]] * mat2x2[[1, 1]] - mat2x2[[0, 1]] * mat2x2[[1, 0]]
}

pub struct DunavantIntegrator {
    integration_order: usize,
}

impl DunavantIntegrator {
    pub fn new(order: usize) -> Self {
        Self {
            integration_order: order,
        }
    }
    fn get_integration_points_multiplicativity(&self) -> Array1<usize> {
        match self.integration_order {
            1 => {
                array![1]
            }
            2 => {
                array![3]
            }
            3 => {
                array![1, 3]
            }
            _ => {
                unimplemented!("See the corresponding paper")
            }
        }
    }
    fn get_integration_points(&self) -> Array2<f64> {
        match self.integration_order {
            1 => {
                array![[
                    0.333_333_333_333_333,
                    0.333_333_333_333_333,
                    1. - 2. * 0.333_333_333_333_333
                ]]
            }
            2 => {
                array![[
                    0.666_666_666_666_667,
                    0.166_666_666_666_667,
                    1. - 0.666_666_666_666_667 - 0.166_666_666_666_667
                ]]
            }
            3 => {
                array![
                    [
                        0.333_333_333_333_333,
                        0.333_333_333_333_333,
                        1. - 2. * 0.333_333_333_333_333
                    ],
                    [0.6, 0.2, 0.2]
                ]
            }
            _ => {
                unimplemented!("See the corresponding paper")
            }
        }
    }
    fn get_integration_weights(&self) -> Array1<f64> {
        match self.integration_order {
            1 => {
                array![1.]
            }
            2 => {
                array![0.333_333_333_333_333]
            }
            3 => {
                array![-0.562_5, 0.520_833_333_333_333]
            }
            _ => {
                unimplemented!("See the corresponding paper")
            }
        }
    }
}

impl Simplex2DIntegrator<IntegratorDummy> for DunavantIntegrator {
    fn dupe(&self) -> Self {
        Self {
            integration_order: self.integration_order,
        }
    }

    fn integrate_over_domain<T: Simplex2DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex2D,
        _cache_data: &mut IntegratorDummy,
    ) -> f64 {
        //let order = self.integration_order;
        let points = self.get_integration_points();
        let multiplicativities = self.get_integration_points_multiplicativity();
        let weights = self.get_integration_weights();

        let num_points = points.nrows();

        let jacobi = array![[1., 0.], [0., 1.], [-1., -1.]];

        let real_jacobi = simplex.get_points().dot(transformation);
        let real_jacobi = real_jacobi.dot(&jacobi);
        let determinant = 0.5 * det2x2(&real_jacobi);
        let mut result = 0.;
        for i in 0..num_points {
            let point = points.row(i);
            let multiplicativity = multiplicativities[i];
            let weight = weights[i];

            match multiplicativity {
                1 => {
                    let integration_point = transformation.dot(&point);
                    result += determinant * weight * func.function_vec(&integration_point, simplex);
                }
                3 => {
                    let perm_points = array![
                        [point[0], point[1], point[2]],
                        [point[2], point[0], point[1]],
                        [point[1], point[2], point[0]]
                    ];
                    let integration_points = transformation.dot(&perm_points);
                    for integration_point in integration_points.columns() {
                        result += determinant
                            * weight
                            * func.function_vec(&integration_point.into_owned(), simplex);
                    }
                }
                6 => {
                    let perm_points = array![
                        [point[0], point[1], point[2]],
                        [point[1], point[0], point[2]],
                        [point[2], point[1], point[0]],
                        [point[1], point[2], point[0]],
                        [point[0], point[2], point[1]],
                        [point[2], point[0], point[1]]
                    ];
                    let integration_points = transformation.dot(&perm_points);
                    for integration_point in integration_points.columns() {
                        result += determinant
                            * weight
                            * func.function_vec(&integration_point.into_owned(), simplex);
                    }
                }
                _ => {
                    panic!()
                }
            }
        }
        return result;
    }
}

#[cfg(test)]
mod tests {
    use crate::integration_2d::integrators::DunavantIntegrator;
    use crate::{integration_2d::domain::IntegratorDummy, integrator_tests};

    integrator_tests! {
        order1: DunavantIntegrator: DunavantIntegrator::new(1), IntegratorDummy: IntegratorDummy::get(),
        order2: DunavantIntegrator: DunavantIntegrator::new(2), IntegratorDummy: IntegratorDummy::get(),
        order3: DunavantIntegrator: DunavantIntegrator::new(2), IntegratorDummy: IntegratorDummy::get(),
    }
}
