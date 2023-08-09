use ndarray::{array, Array1, Array2, concatenate, stack};
use num_dual::DualNum;
use ndarray::Axis;
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

fn det4x4(mat4x4: &Array2<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        let val = mat4x4[[i,0]];
        let mut mat3x3 = Array2::<f64>::zeros([3,3]);
        for j in 0..3 {
            if j >= i {
                for k in 0..3 {
                    mat3x3[[j,k]] = mat4x4[[j+1,k+1]]
                }
            } else {
                for k in 0..3 {
                    mat3x3[[j,k]] = mat4x4[[j,k+1]]
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

        let mut po = Array2::zeros([3,3]);

        for i in 0..3 {
            po[[i,0]] = v1p[i];
            po[[i,1]] = v2p[i];
            po[[i,2]] = v3p[i];
        }
        

        //println!("{}",po);
        return 1./6. * det3x3(&po);
    }
}

/// A general trait implemented by types which supply a function to integrate over.
/// Inputs must be expressed in barycentric coordinates.
pub trait Simplex3DFunction {
    /// The function over the Simplex.
    fn function(&self, xi1: f64, xi2: f64, xi3: f64, xi4: f64, simplex: &Simplex3D) -> f64;

    fn function_vec(&self, xi: &Array1<f64>, simplex: &Simplex3D) -> f64 {
        self.function(xi[0], xi[1], xi[2], xi[3], simplex)
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
    ) -> f64 {
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
    ) -> f64;
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