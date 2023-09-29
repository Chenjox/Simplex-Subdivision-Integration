use ndarray::{array, Array1, Array2};

use crate::integration_3d::{domain::{Simplex3D, Simplex3DFunction, Simplex3DIntegrator}, Simplex3DResultType};

fn det2x2(mat2x2: &Array2<f64>) -> f64 {
    mat2x2[[0, 0]] * mat2x2[[1, 1]] - mat2x2[[0, 1]] * mat2x2[[1, 0]]
}

fn det3x3(mat3x3: &Array2<f64>) -> f64 {
    let mut sum = 0.0;
    for j in 0..3 {
        let mut prod = 1.0;
        for i in 0..3 {
            prod = prod * mat3x3[[i, (i + j) % 3]]
        }
        sum = sum + prod;
        let mut prod = -1.0;
        for i in 0..3 {
            prod = prod * mat3x3[[(2 - i), (i + j) % 3]]
        }
        sum = sum + prod;
    }
    return sum;
}

type Point3D = Array1<f64>;

fn g1(coord: f64) -> f64 {
    if coord > 1.0 || coord < -1.0 {
        panic!("g1() argument out of bounds, !( -1.0 <= {} <= 1.0 )", coord)
    };
    coord / 2.0 + 0.5
}
fn g2(coord: f64) -> f64 {
    if coord > 1.0 || coord < -1.0 {
        panic!("g2() argument out of bounds, !( -1.0 <= {} <= 1.0 )", coord)
    };
    -coord / 2.0 + 0.5
}

/// Interpolation Function G1
#[allow(non_snake_case)]
fn G1(x: f64, y: f64, z: f64) -> f64 {
    g2(x) * g2(y) * g2(z)
}

/// Interpolation Function G2
#[allow(non_snake_case)]
fn G2(x: f64, y: f64, z: f64) -> f64 {
    g1(x) * g2(y) * g2(z)
}

/// Interpolation Function G3
#[allow(non_snake_case)]
fn G3(x: f64, y: f64, z: f64) -> f64 {
    g1(x) * g1(y) * g2(z)
}

/// Interpolation Function G4
#[allow(non_snake_case)]
fn G4(x: f64, y: f64, z: f64) -> f64 {
    g2(x) * g1(y) * g2(z)
}

/// Interpolation Function G5
#[allow(non_snake_case)]
fn G5(x: f64, y: f64, z: f64) -> f64 {
    g2(x) * g2(y) * g1(z)
}

/// Interpolation Function G6
#[allow(non_snake_case)]
fn G6(x: f64, y: f64, z: f64) -> f64 {
    g1(x) * g2(y) * g1(z)
}

/// Interpolation Function G7
#[allow(non_snake_case)]
fn G7(x: f64, y: f64, z: f64) -> f64 {
    g1(x) * g1(y) * g1(z)
}

/// Interpolation Function G8
#[allow(non_snake_case)]
fn G8(x: f64, y: f64, z: f64) -> f64 {
    g2(x) * g1(y) * g1(z)
}

// Help me...

/// Partial Derivative of G1 in X
#[allow(non_snake_case)]
fn G1_diff_x(_x: f64, y: f64, z: f64) -> f64 {
    -0.5 * g2(y) * g2(z)
}

/// Partial Derivative of G1 in Y
#[allow(non_snake_case)]
fn G1_diff_y(x: f64, _y: f64, z: f64) -> f64 {
    g2(x) * (-0.5) * g2(z)
}

/// Partial Derivative of G1 in Z
#[allow(non_snake_case)]
fn G1_diff_z(x: f64, y: f64, _z: f64) -> f64 {
    g2(x) * g2(y) * (-0.5)
}

/// Partial Derivative of G2 in X
#[allow(non_snake_case)]
fn G2_diff_x(_x: f64, y: f64, z: f64) -> f64 {
    0.5 * g2(y) * g2(z)
}

/// Partial Derivative of G2 in Y
#[allow(non_snake_case)]
fn G2_diff_y(x: f64, _y: f64, z: f64) -> f64 {
    g1(x) * (-0.5) * g2(z)
}

/// Partial Derivative of G2 in Z
#[allow(non_snake_case)]
fn G2_diff_z(x: f64, y: f64, _z: f64) -> f64 {
    g1(x) * g2(y) * (-0.5)
}

/// Partial Derivative of G3 in X
#[allow(non_snake_case)]
fn G3_diff_x(_x: f64, y: f64, z: f64) -> f64 {
    0.5 * g1(y) * g2(z)
}

/// Partial Derivative of G3 in Y
#[allow(non_snake_case)]
fn G3_diff_y(x: f64, _y: f64, z: f64) -> f64 {
    g1(x) * 0.5 * g2(z)
}

/// Partial Derivative of G3 in Z
#[allow(non_snake_case)]
fn G3_diff_z(x: f64, y: f64, _z: f64) -> f64 {
    g1(x) * g1(y) * (-0.5)
}

/// Partial Derivative of G4 in X
#[allow(non_snake_case)]
fn G4_diff_x(_x: f64, y: f64, z: f64) -> f64 {
    (-0.5) * g1(y) * g2(z)
}

/// Partial Derivative of G4 in Y
#[allow(non_snake_case)]
fn G4_diff_y(x: f64, _y: f64, z: f64) -> f64 {
    g2(x) * 0.5 * g2(z)
}

/// Partial Derivative of G4 in Z
#[allow(non_snake_case)]
fn G4_diff_z(x: f64, y: f64, _z: f64) -> f64 {
    g2(x) * g1(y) * (-0.5)
}

/// Partial Derivative of G5 in X
#[allow(non_snake_case)]
fn G5_diff_x(_x: f64, y: f64, z: f64) -> f64 {
    (-0.5) * g2(y) * g1(z)
}

/// Partial Derivative of G5 in Y
#[allow(non_snake_case)]
fn G5_diff_y(x: f64, _y: f64, z: f64) -> f64 {
    g2(x) * (-0.5) * g1(z)
}

/// Partial Derivative of G5 in Z
#[allow(non_snake_case)]
fn G5_diff_z(x: f64, y: f64, _z: f64) -> f64 {
    g2(x) * g2(y) * (0.5)
}

/// Partial Derivative of [`G6()`] in X
#[allow(non_snake_case)]
fn G6_diff_x(_x: f64, y: f64, z: f64) -> f64 {
    0.5 * g2(y) * g1(z)
}

/// Partial Derivative of [`G6()`] in Y
#[allow(non_snake_case)]
fn G6_diff_y(x: f64, _y: f64, z: f64) -> f64 {
    g1(x) * (-0.5) * g1(z)
}

/// Partial Derivative of [`G6()`] in Z
#[allow(non_snake_case)]
fn G6_diff_z(x: f64, y: f64, _z: f64) -> f64 {
    g1(x) * g2(y) * 0.5
}

/// Partial Derivative of [`G7()`] in X
#[allow(non_snake_case)]
fn G7_diff_x(_x: f64, y: f64, z: f64) -> f64 {
    0.5 * g1(y) * g1(z)
}

/// Partial Derivative of [`G7()`] in Y
#[allow(non_snake_case)]
fn G7_diff_y(x: f64, _y: f64, z: f64) -> f64 {
    g1(x) * 0.5 * g1(z)
}

/// Partial Derivative of [`G7()`] in Z
#[allow(non_snake_case)]
fn G7_diff_z(x: f64, y: f64, _z: f64) -> f64 {
    g1(x) * g1(y) * 0.5
}

/// Partial Derivative of [`G8()`] in X
#[allow(non_snake_case)]
fn G8_diff_x(_x: f64, y: f64, z: f64) -> f64 {
    (-0.5) * g1(y) * g1(z)
}

/// Partial Derivative of [`G8()`] in Y
#[allow(non_snake_case)]
fn G8_diff_y(x: f64, _y: f64, z: f64) -> f64 {
    g2(x) * 0.5 * g1(z)
}

/// Partial Derivative of [`G8()`] in Z
#[allow(non_snake_case)]
fn G8_diff_z(x: f64, y: f64, _z: f64) -> f64 {
    g2(x) * g1(y) * 0.5
}

/// Linear Interpolation function from [-1,1] to [p1,p2]
fn interpolation_function_vector(xyz1: Point3D) -> Array1<f64> {
    let x = xyz1[0];
    let y = xyz1[1];
    let z = xyz1[2];
    return array![
        G1(x, y, z),
        G2(x, y, z),
        G3(x, y, z),
        G4(x, y, z),
        G5(x, y, z),
        G6(x, y, z),
        G7(x, y, z),
        G8(x, y, z)
    ];
}

/// Linear Interpolation function derivates from [-1,1] to [p1,p2]
fn interpolation_function_derivatives_vector(xyz1: Point3D) -> Array2<f64> {
    let x = xyz1[0];
    let y = xyz1[1];
    let z = xyz1[2];
    return array![
        [G1_diff_x(x, y, z), G1_diff_y(x, y, z), G1_diff_z(x, y, z)],
        [G2_diff_x(x, y, z), G2_diff_y(x, y, z), G2_diff_z(x, y, z)],
        [G3_diff_x(x, y, z), G3_diff_y(x, y, z), G3_diff_z(x, y, z)],
        [G4_diff_x(x, y, z), G4_diff_y(x, y, z), G4_diff_z(x, y, z)],
        [G5_diff_x(x, y, z), G5_diff_y(x, y, z), G5_diff_z(x, y, z)],
        [G6_diff_x(x, y, z), G6_diff_y(x, y, z), G6_diff_z(x, y, z)],
        [G7_diff_x(x, y, z), G7_diff_y(x, y, z), G7_diff_z(x, y, z)],
        [G8_diff_x(x, y, z), G8_diff_y(x, y, z), G8_diff_z(x, y, z)]
    ];
}

pub struct Quadrilateral3DIntegrator {
    gauss_degree: usize,
}

impl Quadrilateral3DIntegrator {
    pub fn new(gauss_degree: usize) -> Self {
        Self { gauss_degree }
    }

    fn get_gauss_points(&self) -> Array1<f64> {
        match self.gauss_degree {
            1 => {
                array![0.]
            }
            2 => {
                array![-(1. / 3.0_f64).sqrt(), (1. / 3.0_f64).sqrt()]
            }
            3 => {
                array![-(3. / 5.0_f64).sqrt(), 0., (3. / 5.0_f64).sqrt()]
            }
            _ => {
                unimplemented!("For higher order gaussian Integration the Golub Welsh Algorithm should be implemented.")
            }
        }
    }
    fn get_gauss_weights(&self) -> Array1<f64> {
        match self.gauss_degree {
            1 => {
                array![2.]
            }
            2 => {
                array![1., 1.]
            }
            3 => {
                array![5. / 9., 8. / 9., 5. / 9.]
            }
            _ => {
                unimplemented!("For higher order gaussian Integration the Golub Welsh Algorithm should be implemented.")
            }
        }
    }
    fn get_quadrilateral(index: usize) -> Array2<f64> {
        match index {
            1 => {
                array![
                    [1., 0.5, 1. / 3., 0.5, 0.5, 1. / 3., 0.25, 1. / 3.],
                    [0., 0.5, 1. / 3., 0., 0., 1. / 3., 0.25, 0.],
                    [0., 0., 1. / 3., 0.5, 0., 0., 0.25, 1. / 3.],
                    [0., 0., 0., 0., 0.5, 1. / 3., 0.25, 1. / 3.]
                ]
            }
            2 => {
                array![
                    [0., 0., 1. / 3., 0.5, 0., 0., 0.25, 1. / 3.],
                    [1., 0.5, 1. / 3., 0.5, 0.5, 1. / 3., 0.25, 1. / 3.],
                    [0., 0.5, 1. / 3., 0., 0., 1. / 3., 0.25, 0.],
                    [0., 0., 0., 0., 0.5, 1. / 3., 0.25, 1. / 3.]
                ]
            }
            3 => {
                array![
                    [0., 0.5, 1. / 3., 0., 0., 1. / 3., 0.25, 0.],
                    [0., 0., 1. / 3., 0.5, 0., 0., 0.25, 1. / 3.],
                    [1., 0.5, 1. / 3., 0.5, 0.5, 1. / 3., 0.25, 1. / 3.],
                    [0., 0., 0., 0., 0.5, 1. / 3., 0.25, 1. / 3.]
                ]
            }
            4 => {
                array![
                    [0., 0.5, 1. / 3., 0., 0., 1. / 3., 0.25, 0.],
                    [0., 0., 0., 0., 0.5, 1. / 3., 0.25, 1. / 3.],
                    [0., 0., 1. / 3., 0.5, 0., 0., 0.25, 1. / 3.],
                    [1., 0.5, 1. / 3., 0.5, 0.5, 1. / 3., 0.25, 1. / 3.]
                ]
            }
            _ => {
                panic!("Illegal Domain chosen!")
            }
        }
    }

    fn integrate_quadrilateral<T: Simplex3DFunction>(
        &self,
        barycentric_domain: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex3D,
    ) -> T::Return {
        let mut sum = func.additive_neutral_element();
        let gauss_points = self.get_gauss_points();
        let gauss_weights = self.get_gauss_weights();
        for i in 0..self.gauss_degree {
            for j in 0..self.gauss_degree {
                for k in 0..self.gauss_degree {
                    let X = gauss_points[i];
                    let Y = gauss_points[j];
                    let Z = gauss_points[k];
                    // M(8,1)
                    let inte_space_interpolation = interpolation_function_vector(array![X, Y, Z]);
                    // M(8,3)
                    let inte_space_jacobi =
                        interpolation_function_derivatives_vector(array![X, Y, Z]);

                    // M(4,1) = M(4,8) x M(8,1)
                    let barycentric_coords = barycentric_domain.dot(&inte_space_interpolation);
                    // M(4,3) = M(4,8) x M(8,3)
                    let barycentric_jacobi = barycentric_domain.dot(&inte_space_jacobi);

                    // M(3,3) = M(3,4) x M(4,3)
                    let real_jacobi = simplex.get_points().dot(&barycentric_jacobi);
                    let determinant = det3x3(&real_jacobi);

                    let mut func_result = func.function_vec(&barycentric_coords, simplex);

                    func_result *= determinant * gauss_weights[i]
                        * gauss_weights[j]
                        * gauss_weights[k];

                    sum.add_assign(&func_result);
                }
            }
        }

        return sum;
    }
}

impl<IntegratorDummy> Simplex3DIntegrator<IntegratorDummy> for Quadrilateral3DIntegrator {
    fn integrate_over_domain<T: Simplex3DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex3D,
        _cached_data: &mut IntegratorDummy,
    ) -> T::Return {
        if !(transformation.shape()[0] == 4 && transformation.shape()[1] == 4) {
            panic!(
                "Die Transformationsmatrix ist nicht der Dimension 4 x 4, sondern {} x {}",
                transformation.shape()[0],
                transformation.shape()[1]
            )
        }
        let mut sum = func.additive_neutral_element();
        for i in 1..=4 {
            // M(4,8)
            let d1 = Quadrilateral3DIntegrator::get_quadrilateral(i);
            // M(4,8) = M(4,4) x M(4,8)
            let d1 = transformation.dot(&d1);
            sum.add_assign(&self.integrate_quadrilateral(&d1, func, simplex));
        }
        return sum;
    }
}
