use std::ops::{AddAssign, MulAssign};

use ndarray::Axis;
use ndarray::{array, concatenate, stack, Array1, Array2};
use num_dual::DualNum;
pub struct Simplex3D {
    points: Array2<f64>,
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

pub fn det4x4(mat4x4: &Array2<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        let val = mat4x4[[i, 0]];
        let mut mat3x3 = Array2::<f64>::zeros([3, 3]);
        for j in 0..3 {
            if j >= i {
                for k in 0..3 {
                    mat3x3[[j, k]] = mat4x4[[j + 1, k + 1]]
                }
            } else {
                for k in 0..3 {
                    mat3x3[[j, k]] = mat4x4[[j, k + 1]]
                }
            }
        }
        sum += (-1.).powi(i as i32) * val * det3x3(&mat3x3);
    }
    return sum;
}

type Point3D = Array1<f64>;

impl Simplex3D {
    pub fn new_from_points(p1: &Point3D, p2: &Point3D, p3: &Point3D, p4: &Point3D) -> Self {
        Self {
            points: array![
                [p1[0], p2[0], p3[0], p4[0]],
                [p1[1], p2[1], p3[1], p4[1]],
                [p1[2], p2[2], p3[2], p4[2]]
            ],
        }
    }

    pub fn new_from_array(points: Array2<f64>) -> Self {
        Self { points }
    }

    pub fn get_points(&self) -> Array2<f64> {
        return self.points.clone();
    }

    pub fn get_volume(&self) -> f64 {
        let po = &self.points;
        let v0 = po.column(0);
        let v1 = po.column(1);
        let v2 = po.column(2);
        let v3 = po.column(3);

        let v1p = &v1 - &v0;
        let v2p = &v2 - &v0;
        let v3p = &v3 - &v0;

        let mut po = Array2::zeros([3, 3]);

        for i in 0..3 {
            po[[i, 0]] = v1p[i];
            po[[i, 1]] = v2p[i];
            po[[i, 2]] = v3p[i];
        }

        //println!("{}",po);
        return 1. / 6. * det3x3(&po);
    }
}

pub trait Simplex3DResultType: MulAssign<f64> + AddAssign<f64> {
    fn add_assign(&mut self, other: &Self);

    fn distance(&self, other: &Self) -> f64;

    fn additive_neutral_element() -> Self;
}

#[derive(Debug)]
pub struct ResultTypeWrapper<T>(T);

impl<T> ResultTypeWrapper<T> {
    pub fn new(t: T) -> Self {
        Self(t)
    }

    pub fn get(self) -> T {
        self.0
    }

    pub fn get_borrow(&self) -> &T {
        &self.0
    }
}

impl MulAssign<f64> for ResultTypeWrapper<f64> {
    fn mul_assign(&mut self, rhs: f64) {
        self.0 *= rhs
    }
}

impl AddAssign<f64> for ResultTypeWrapper<f64> {
    fn add_assign(&mut self, rhs: f64) {
        self.0 += rhs
    }
}

impl Simplex3DResultType for ResultTypeWrapper<f64> {
    fn add_assign(&mut self, other: &Self) {
        self.0 += other.0
    }

    fn distance(&self, other: &Self) -> f64 {
        (self.0 - other.0).abs()
    }

    fn additive_neutral_element() -> Self {
        Self(0.)
    }
}

impl MulAssign<f64> for ResultTypeWrapper<Array2<f64>> {
    fn mul_assign(&mut self, rhs: f64) {
        self.0 *= rhs
    }
}

impl AddAssign<f64> for ResultTypeWrapper<Array2<f64>> {
    fn add_assign(&mut self, rhs: f64) {
        self.0 += rhs
    }
}

impl Simplex3DResultType for ResultTypeWrapper<Array2<f64>> {
    fn add_assign(&mut self, other: &Self) {
        self.0 = &self.0 + &other.0
    }

    fn distance(&self, other: &Self) -> f64 {
        let diff = &self.0 - &other.0;
        diff.iter().map(|f| f.powi(2)).sum::<f64>().sqrt()
    }

    fn additive_neutral_element() -> Self {
        Self(Array2::zeros([10, 10]))
    }
}

/// A general trait implemented by types which supply a function to integrate over.
/// Inputs must be expressed in barycentric coordinates.
pub trait Simplex3DFunction {
    type Return: Simplex3DResultType;
    /// The function over the Simplex.
    fn function(&self, xi1: f64, xi2: f64, xi3: f64, xi4: f64, simplex: &Simplex3D)
        -> Self::Return;

    fn function_vec(&self, xi: &Array1<f64>, simplex: &Simplex3D) -> Self::Return {
        self.function(xi[0], xi[1], xi[2], xi[3], simplex)
    }

    fn additive_neutral_element(&self) -> Self::Return {
        Self::Return::additive_neutral_element()
    }
}

/// A general trait implemented by types which supply an integration scheme for a single Simplex.
/// Allows for easy substitution of simplex integration schemes.
pub trait Simplex3DIntegrator<D> {
    /// This function will be called on a single simplex, given in the third argument.
    fn integrate_simplex<T: Simplex3DFunction>(
        &self,
        func: &Box<T>,
        simplex: &Simplex3D,
        cache_data: &mut D,
    ) -> T::Return {
        self.integrate_over_domain(
            &array![
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            func,
            simplex,
            cache_data,
        )
    }

    /// A more general function which takes a transformation matrix to map to the initial subdomain of the simplex
    fn integrate_over_domain<T: Simplex3DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex3D,
        cache_data: &mut D,
    ) -> T::Return;
}

//fn usage(sim: &Simplex2D, func: &Box<dyn Simplex2DFunction>, inte: &Box<dyn Simplex2DIntegrator>) {
//    let val = inte.integrate(func, sim);
//}
#[macro_export]
macro_rules! integrator_tests_3d {
    ($($name:ident: $type:ty: $init:expr, $typecache:ty: $initcache:expr,)*) => {
    $(
        mod $name {
            use super::*;
            use crate::integration_3d::functions::*;
            use crate::integration_3d::domain::*;
            use ndarray::array;

            fn get_integrator_cache() -> $typecache {
                $initcache
            }

            fn get_instance() -> $type {
                $init
            }

            #[test]
            fn constant_function_correctness_abs() {
                let inte = get_instance();
                let mut cache = get_integrator_cache();

                let func = Box::new(Constant3DFunction {});
                // Gegen den Uhrzeigersinn
                let sim = Simplex3D::new_from_points(
                    &array![ (8./9.0_f64).sqrt() , 0., - 1./3.],
                    &array![-(2./9.0_f64).sqrt(), (2./3.0_f64).sqrt(),-1./3.],
                    &array![-(2./9.0_f64).sqrt(),-(2./3.0_f64).sqrt(),-1./3.],
                    &array![0.,0.,1.],
                );

                

                let result = inte.integrate_simplex(&func, &sim, &mut cache);

                let true_result = ResultTypeWrapper::new((2.0_f64).sqrt() * ((8./3.0_f64).sqrt()).powi(3) / 12.);
                let approx_eq = true_result.distance(&result);

                assert!(approx_eq <= 1e-2_f64, "Expected: {:?}, Actual: {:?}, Diff: {}",true_result, result, approx_eq);
            }

            #[test]
            fn constant_function_correctness() {
                let inte = get_instance();
                let mut cache = get_integrator_cache();

                let func = Box::new(Constant3DFunction {});
                // Gegen den Uhrzeigersinn
                let sim = Simplex3D::new_from_points(
                    &array![ (8./9.0_f64).sqrt() , 0., - 1./3.],
                    &array![-(2./9.0_f64).sqrt(), (2./3.0_f64).sqrt(),-1./3.],
                    &array![-(2./9.0_f64).sqrt(),-(2./3.0_f64).sqrt(),-1./3.],
                    &array![0.,0.,1.],
                );

                let result = inte.integrate_simplex(&func, &sim, &mut cache);

                let true_result = ResultTypeWrapper::new((2.0_f64).sqrt() * ((8./3.0_f64).sqrt()).powi(3) / 12.);
                let approx_eq = true_result.distance(&result);

                assert!(approx_eq <= 1e-2_f64, "Expected: {:?}, Actual: {:?}, Diff: {}",true_result, result, approx_eq);
            }

            #[test]
            fn preserving_orientation() {
                let inte = get_instance();
                let mut cache = get_integrator_cache();

                let func = Box::new(Constant3DFunction {});
                // Gegen den Uhrzeigersinn
                let sim = Simplex3D::new_from_points(
                    &array![ (8./9.0_f64).sqrt() , 0., - 1./3.],
                    &array![-(2./9.0_f64).sqrt(), (2./3.0_f64).sqrt(),-1./3.],
                    &array![-(2./9.0_f64).sqrt(),-(2./3.0_f64).sqrt(),-1./3.],
                    &array![0.,0.,1.],
                );


                let result = inte.integrate_simplex(&func, &sim, &mut cache);

                let true_result = ResultTypeWrapper::new((2.0_f64).sqrt() * ((8./3.0_f64).sqrt()).powi(3) / 12.);
                let approx_eq = result.get_borrow().signum() + true_result.get_borrow().signum();

                assert!(approx_eq == 2.0, "Expected: {:?}, Actual: {:?}",result,true_result);
            }
        }
    )*
    }
}