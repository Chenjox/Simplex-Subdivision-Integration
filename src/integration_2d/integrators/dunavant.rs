use ndarray::{array, Array1, Array2};

use crate::integration_2d::domain::{Simplex2D, Simplex2DFunction, Simplex2DIntegrator};

fn det2x2(mat2x2: &Array2<f64>) -> f64 {
    mat2x2[[0, 0]] * mat2x2[[1, 1]] - mat2x2[[0, 1]] * mat2x2[[1, 0]]
}

pub struct DunavantIntegrator {
    gauss_degree: usize,
}

impl DunavantIntegrator {
    fn get_integration_points_multiplicativity(&self) -> Array1<usize> {
        match self.gauss_degree {
            1 => {
                array![1]
            }
            _ => {
                unimplemented!("See the corresponding paper")
            }
        }
    }
    fn get_integration_points(&self) -> Array2<f64> {
        match self.gauss_degree {
            1 => {
                array![[
                    0.333_333_333_333_333,
                    0.333_333_333_333_333,
                    1. - 2. * 0.333_333_333_333_333
                ]]
            }
            _ => {
                unimplemented!("See the corresponding paper")
            }
        }
    }
    fn get_integration_weights(&self) -> Array1<f64> {
        match self.gauss_degree {
            1 => {
                array![1.]
            }
            _ => {
                unimplemented!("See the corresponding paper")
            }
        }
    }
}
