use ndarray::{array, Array1, Array2};

type Point2D = Array1<f64>;

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
}

/// A general trait implemented by types which supply a function to integrate over.
/// Inputs must be expressed in barycentric coordinates.
pub trait Simplex2DFunction {
    /// The function over the Simplex.
    fn function(&self, xi1: f64, xi2: f64, xi3: f64, simplex: &Simplex2D) -> f64;

    fn function_vec(&self, xi: &Array1<f64>, simplex: &Simplex2D) -> f64 {
        self.function(xi[0], xi[1], xi[2], simplex)
    }
}

/// A general trait implemented by types which supply an integration scheme for a single Simplex.
/// Allows for easy substitution of simplex integration schemes.
pub trait Simplex2DIntegrator<D> {
    /// This function will be called on a single simplex, given in the third argument.
    fn integrate_simplex<T: Simplex2DFunction>(
        &self,
        func: &Box<T>,
        simplex: &Simplex2D,
        cache_data: &mut D,
    ) -> f64 {
        self.integrate_over_domain(
            &array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            func,
            simplex,
            cache_data,
        )
    }

    /// A more general function which takes a transformation matrix to map to the initial subdomain of the simplex
    fn integrate_over_domain<T: Simplex2DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex2D,
        cache_data: &mut D,
    ) -> f64;
}

pub type PureSimplex2DIntegrator = dyn Simplex2DIntegrator<IntegratorDummy>;

pub struct IntegratorDummy;

impl IntegratorDummy {
    pub fn get() -> Self {
        Self {}
    }
}

//fn usage(sim: &Simplex2D, func: &Box<dyn Simplex2DFunction>, inte: &Box<dyn Simplex2DIntegrator>) {
//    let val = inte.integrate(func, sim);
//}
