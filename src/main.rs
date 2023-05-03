use indextree::Arena;
use ndarray::prelude::*;

use crate::domain::*;

mod domain;
mod problems;

type Point2D = Array1<f64>;
type Point3D = Array1<f64>;

type PointMatrix = Array2<f64>;
/// Poor programmers solution for determinant
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

fn G1(x1: f64, x2: f64) -> f64 {
    g1(x1) * g1(x2)
}

fn G1_diff_x1(x1: f64, x2: f64) -> f64 {
    0.5 * g1(x2)
}
fn G1_diff_x2(x1: f64, x2: f64) -> f64 {
    g1(x1) * 0.5
}

fn G2(x1: f64, x2: f64) -> f64 {
    g1(x1) * g2(x2)
}

fn G2_diff_x1(x1: f64, x2: f64) -> f64 {
    0.5 * g2(x2)
}
fn G2_diff_x2(x1: f64, x2: f64) -> f64 {
    g1(x1) * -0.5
}

fn G3(x1: f64, x2: f64) -> f64 {
    g2(x1) * g1(x2)
}

fn G3_diff_x1(x1: f64, x2: f64) -> f64 {
    -0.5 * g1(x2)
}
fn G3_diff_x2(x1: f64, x2: f64) -> f64 {
    g2(x1) * 0.5
}

fn G4(x1: f64, x2: f64) -> f64 {
    g2(x1) * g2(x2)
}

fn G4_diff_x1(x1: f64, x2: f64) -> f64 {
    -0.5 * g2(x2)
}
fn G4_diff_x2(x1: f64, x2: f64) -> f64 {
    g2(x1) * -0.5
}

/// Linear Interpolation function from [-1,1] to [p1,p2]
fn interpolation_function_vector(xy1: Point2D) -> Array1<f64> {
    let x1 = xy1[0];
    let x2 = xy1[1];
    return array![G1(x1, x2), G2(x1, x2), G3(x1, x2), G4(x1, x2)];
}

/// Linear Interpolation function derivates from [-1,1] to [p1,p2]
fn interpolation_function_derivatives_vector(xy1: Point2D) -> Array2<f64> {
    let x1 = xy1[0];
    let x2 = xy1[1];
    return array![
        [G1_diff_x1(x1, x2), G1_diff_x2(x1, x2)],
        [G2_diff_x1(x1, x2), G2_diff_x2(x1, x2)],
        [G3_diff_x1(x1, x2), G3_diff_x2(x1, x2)],
        [G4_diff_x1(x1, x2), G4_diff_x2(x1, x2)]
    ];
}

pub struct Quadrilateral2DIntegration {
    gauss_degree: usize,
}

impl Quadrilateral2DIntegration {
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
    fn get_quadrilateral_D1() -> Array2<f64> {
        // B1 B5 B4 BC
        return array![
            [1., 0.5, 0.5, 1. / 3.],
            [0., 0.5, 0., 1. / 3.],
            [0., 0., 0.5, 1. / 3.]
        ];
    }
    fn get_quadrilateral_D2() -> Array2<f64> {
        // B2 B6 B5 BC
        return array![
            [0., 0., 0.5, 1. / 3.],
            [1., 0.5, 0.5, 1. / 3.],
            [0., 0.5, 0., 1. / 3.]
        ];
    }
    fn get_quadrilateral_D3() -> Array2<f64> {
        // B3 B4 B6 BC
        return array![
            [0., 0.5, 0., 1. / 3.],
            [0., 0., 0.5, 1. / 3.],
            [1., 0.5, 0.5, 1. / 3.]
        ];
    }

    fn integrate_quadrilateral<T : Simplex2DFunction>(
        &self,
        barycentric_domain: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex2D,
    ) -> f64 {
        let mut sum = 0.0;
        let gauss_points = self.get_gauss_points();
        let gauss_weights = self.get_gauss_weights();
        for i in 0..self.gauss_degree {
            for j in 0..self.gauss_degree {
                let X = gauss_points[i];
                let Y = gauss_points[j];
                let inte_space_interpolation = interpolation_function_vector(array![X, Y]);
                let inte_space_jacobi = interpolation_function_derivatives_vector(array![X, Y]);

                let barycentric_coords = barycentric_domain.dot(&inte_space_interpolation);
                let barycentric_jacobi = barycentric_domain.dot(&inte_space_jacobi);

                let real_jacobi = simplex.get_points().dot(&barycentric_jacobi);
                let determinant = det2x2(&real_jacobi);

                let func_result = func.function_vec(&barycentric_coords, simplex);

                let integral_result =
                    determinant * func_result * gauss_weights[i] * gauss_weights[j];

                sum += integral_result;
            }
        }

        return sum;
    }
}

impl Simplex2DIntegrator for Quadrilateral2DIntegration {
    fn integrate<T : Simplex2DFunction>(&self, func: &Box<T>, simplex: &Simplex2D) -> f64 {
        let mut sum = 0.;
        let d1 = Quadrilateral2DIntegration::get_quadrilateral_D1();
        sum += self.integrate_quadrilateral(&d1, func, simplex);
        let d2 = Quadrilateral2DIntegration::get_quadrilateral_D2();
        sum += self.integrate_quadrilateral(&d2, func, simplex);
        let d3 = Quadrilateral2DIntegration::get_quadrilateral_D3();
        sum += self.integrate_quadrilateral(&d3, func, simplex);

        return sum;
    }
}

struct Hierarchic2DIntegration<I : Simplex2DIntegrator> {
    base_integrator: I
}

impl<I : Simplex2DIntegrator> Hierarchic2DIntegration<I> {

}

impl<I : Simplex2DIntegrator> Simplex2DIntegrator for Hierarchic2DIntegration<I> {
    fn integrate<T : Simplex2DFunction>(&self, func: &Box<T>, simplex: &Simplex2D) -> f64 {
        // It all begins with a tree!
        
        unimplemented!();
    }
}

fn main() {
    // ASSERTION: The Simplex is always rightly oriented.
    let sim = Simplex2D::new(&array![1., 1.], &array![1., 2.], &array![3., 1.]);
    let inte = Quadrilateral2DIntegration { gauss_degree: 3 };

    let func = Box::new(Constant2DFunctionHistory::new());

    let result = inte.integrate(&func, &sim);
    println!("{}", result);

    let hist = func.get_history();

    for el in hist {
        //println!("{},{}",el, el.fold(0., |f1, f2| f1 + f2));
        println!("\\draw[fill,red] (barycentric cs:ca={:.3},cb={:.3},cc={:.3}) coordinate (cb1) circle (2pt);",el[0],el[1],el[2]);
    }
    /*
    let tree = &mut Arena::new();

    let a = tree.new_node(1);
    let b = tree.new_node(11);
    let c = tree.new_node(12);
    let d = tree.new_node(13);

    a.append(b, tree);
    a.append(c, tree);
    a.append(d, tree);


    println!("{:?}",tree);

    for node in a.descendants(tree) {
        println!("{}",tree[node].get())
    }
     */
}
