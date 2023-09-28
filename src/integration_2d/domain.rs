use std::ops::{Add, AddAssign, Mul, MulAssign};

use ndarray::{array, Array1, Array2, Axis};

type Point2D = Array1<f64>;

pub fn det3x3(mat3x3: &Array2<f64>) -> f64 {
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

/// A struct representing a simplex on the Euclidean 2D Plane
pub struct Simplex2D {
    points: Array2<f64>,
}

impl Simplex2D {
    pub fn new_from_points(p1: &Point2D, p2: &Point2D, p3: &Point2D) -> Self {
        Self {
            points: array![[p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]]],
        }
    }

    pub fn new_from_array(points: Array2<f64>) -> Self {
        Self { points }
    }

    pub fn get_points(&self) -> Array2<f64> {
        return self.points.clone();
    }

    pub fn get_area(&self) -> f64 {
        let mut points = self.points.clone();
        points
            .append(Axis(0), array![[1.0_f64, 1., 1.]].view())
            .unwrap();

        //let points = points.reversed_axes();
        println!("{}", points);

        return det3x3(&points) / 2.;
    }
}

pub trait Simplex2DResultType: MulAssign<f64> + AddAssign<f64> {
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

impl Simplex2DResultType for ResultTypeWrapper<f64> {
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

impl Simplex2DResultType for ResultTypeWrapper<Array2<f64>> {
    fn add_assign(&mut self, other: &Self) {
        self.0 = &self.0 + &other.0
    }

    fn distance(&self, other: &Self) -> f64 {
        let diff = &self.0 - &other.0;
        diff.iter().map(|f| f.powi(2)).sum::<f64>().sqrt()
    }

    fn additive_neutral_element() -> Self {
        Self(Array2::zeros([6, 6]))
    }
}

/// A general trait implemented by types which supply a function to integrate over.
/// Inputs must be expressed in barycentric coordinates.
pub trait Simplex2DFunction {
    type Return: Simplex2DResultType;

    fn additive_neutral_element(&self) -> Self::Return {
        Self::Return::additive_neutral_element()
    }
    /// The function over the Simplex.
    fn function(&self, xi1: f64, xi2: f64, xi3: f64, simplex: &Simplex2D) -> Self::Return;

    fn function_vec(&self, xi: &Array1<f64>, simplex: &Simplex2D) -> Self::Return {
        self.function(xi[0], xi[1], xi[2], simplex)
    }
}

/// A general trait implemented by types which supply an integration scheme for a single Simplex.
/// Allows for easy substitution of simplex integration schemes.
pub trait Simplex2DIntegrator<D> {
    fn dupe(&self) -> Self;

    /// This function will be called on a single simplex, given in the third argument.
    fn integrate_simplex<T: Simplex2DFunction>(
        self: &Self,
        func: &Box<T>,
        simplex: &Simplex2D,
        cache_data: &mut D,
    ) -> T::Return {
        self.integrate_over_domain(
            &array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            func,
            simplex,
            cache_data,
        )
    }

    /// A more general function which takes a transformation matrix to map to the initial subdomain of the simplex
    fn integrate_over_domain<T: Simplex2DFunction>(
        self: &Self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex2D,
        cache_data: &mut D,
    ) -> T::Return;
}

pub struct IntegratorDummy;

impl IntegratorDummy {
    pub fn get() -> Self {
        Self {}
    }
}

//fn usage(sim: &Simplex2D, func: &Box<dyn Simplex2DFunction>, inte: &Box<dyn Simplex2DIntegrator>) {
//    let val = inte.integrate(func, sim);
//}
#[macro_export]
macro_rules! integrator_tests {
    ($($name:ident: $type:ty: $init:expr, $typecache:ty: $initcache:expr,)*) => {
    $(
        mod $name {
            use super::*;
            use crate::integration_2d::functions::*;
            use crate::integration_2d::domain::*;
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

                let func = Box::new(Constant2DFunction {});
                // Gegen den Uhrzeigersinn
                let sim = Simplex2D::new_from_points(
                    &array![0.0_f64,0.],
                    &array![1.,0.],
                    &array![0.,1.],
                );

                let result = inte.integrate_simplex(&func, &sim, &mut cache);

                let true_result = ResultTypeWrapper::new(0.5_f64);
                let approx_eq = true_result.distance(&result);

                assert!(approx_eq <= 1e-2_f64, "Expected: {:?}, Actual: {:?}, Diff: {}",true_result, result, approx_eq);
            }

            #[test]
            fn constant_function_correctness() {
                let inte = get_instance();
                let mut cache = get_integrator_cache();

                let func = Box::new(Constant2DFunction {});
                // Gegen den Uhrzeigersinn
                let sim = Simplex2D::new_from_points(
                    &array![0.0_f64,0.],
                    &array![1.,0.],
                    &array![0.,1.],
                );

                let result = inte.integrate_simplex(&func, &sim, &mut cache);

                let true_result = ResultTypeWrapper::new(0.5_f64);
                let approx_eq = true_result.distance(&result);

                assert!(approx_eq <= 1e-2_f64, "Expected: {:?}, Actual: {:?}, Diff: {}",true_result, result, approx_eq);
            }

            #[test]
            fn preserving_orientation() {
                let inte = get_instance();
                let mut cache = get_integrator_cache();

                let func = Box::new(Constant2DFunction {});
                // Gegen den Uhrzeigersinn
                let sim = Simplex2D::new_from_points(
                    &array![0.0_f64,0.],
                    &array![1.,0.],
                    &array![0.,1.],
                );

                let result = inte.integrate_simplex(&func, &sim, &mut cache);

                let true_result = ResultTypeWrapper::new(0.5_f64);
                let approx_eq = result.get_borrow().signum() + true_result.get_borrow().signum();

                assert!(approx_eq == 2.0, "Expected: {:?}, Actual: {:?}",result,true_result);
            }
        }
    )*
    }
}
