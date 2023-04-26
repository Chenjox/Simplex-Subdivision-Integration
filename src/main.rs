use indextree::Arena;
use ndarray::prelude::*;

mod problems;
mod domain;

type Point2D = Array1<f64>;
type Point3D = Array1<f64>;

type PointMatrix = Array2<f64>;
/// Poor programmings solution for determinant
fn det2x2(mat2x2: &Array2<f64>)-> f64 {
    mat2x2[[0,0]] * mat2x2[[1,1]] - mat2x2[[0,1]] * mat2x2[[1,0]]
}

fn new_point_2D(x1: f64, x2: f64) -> Point2D {
    let mut point = Point2D::zeros(2);
    point[0] = x1;
    point[1] = x2;
    return  point;
}

fn g1(coord: f64) -> f64 {
    coord/2.0 + 0.5
}
fn g2(coord: f64) -> f64 {
    - coord/2.0 + 0.5
}

fn G1(x1: f64,x2: f64) -> f64 {
    g1(x1)*g1(x2)
}

fn G1_diff_x1(x1: f64,x2: f64) -> f64 { 0.5 * g1(x2)}
fn G1_diff_x2(x1: f64,x2: f64) -> f64 { g1(x1) * 0.5}

fn G2(x1: f64,x2: f64) -> f64 {
    g1(x1)*g2(x2)
}

fn G2_diff_x1(x1: f64,x2: f64) -> f64 { 0.5 * g2(x2)}
fn G2_diff_x2(x1: f64,x2: f64) -> f64 { g1(x1) * -0.5}

fn G3(x1: f64,x2: f64) -> f64 {
    g2(x1)*g1(x2)
}

fn G3_diff_x1(x1: f64,x2: f64) -> f64 { -0.5 * g1(x2)}
fn G3_diff_x2(x1: f64,x2: f64) -> f64 { g2(x1) * 0.5}

fn G4(x1: f64,x2: f64) -> f64 {
    g2(x1)*g2(x2)
}

fn G4_diff_x1(x1: f64,x2: f64) -> f64 { -0.5 * g2(x2)}
fn G4_diff_x2(x1: f64,x2: f64) -> f64 { g2(x1) * -0.5}

/// Linear Interpolation function from [-1,1] to [p1,p2]
fn interpolation_function_vector(xy1: Point2D) -> Array1<f64> {
    let x1 = xy1[0];
    let x2 = xy1[1];
    return array![ G1(x1,x2) , G2(x1,x2), G3(x1,x2), G4(x1,x2) ];
}

/// Linear Interpolation function derivates from [-1,1] to [p1,p2]
fn interpolation_function_derivatives_vector(xy1: Point2D) -> Array2<f64> {
    let x1 = xy1[0];
    let x2 = xy1[1];
    return array![ 
        [G1_diff_x1(x1, x2), G1_diff_x2(x1, x2)],
    [G2_diff_x1(x1, x2), G2_diff_x2(x1, x2)],
    [G3_diff_x1(x1, x2), G3_diff_x2(x1, x2)],
    [G4_diff_x1(x1, x2), G4_diff_x2(x1, x2)]];
}

///Hardcoded Domain
fn domain_2D_matrix() -> Array2<f64> {
    // B1 B5 B4 BC
    return array![ [1. , 0.5 ,0.5 , 1./3.],
    [0., 0.5, 0., 1./3.],
    [0., 0. , 0.5, 1./3.] ];
}

struct Triangle2D {
    p1: Point2D,
    p2: Point2D,
    p3: Point2D
}

impl Triangle2D {
    fn get_barycentric_matrix(&self) -> PointMatrix {
        let mut point_matrix = PointMatrix::zeros((2,3));
        point_matrix[[0,0]] = self.p1[0];
        point_matrix[[1,0]] = self.p1[1];
        point_matrix[[0,1]] = self.p2[0];
        point_matrix[[1,1]] = self.p2[1];
        point_matrix[[0,2]] = self.p3[0];
        point_matrix[[1,2]] = self.p3[1];
        return point_matrix;
    }
    fn get_point_barycentric(&self, xi1: f64, xi2: f64, xi3: f64) -> Point2D {
        if xi1 + xi2 + xi3 > 1.0 {panic!("Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",xi1,xi2,xi3,xi1+xi2+xi3);};
        return xi1 * &self.p1 + xi2 * &self.p2 + xi3 * &self.p3;
    }


}

struct Subdivision {
    p1: Point3D,
    p2: Point3D,
    p3: Point3D
}

impl Subdivision {
    fn get_point_barycentric(self, xi1: f64, xi2: f64, xi3: f64) -> Point3D {
        if xi1 + xi2 + xi3 > 1.0 {panic!("Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",xi1,xi2,xi3,xi1+xi2+xi3);};
        return xi1 * self.p1 + xi2 * self.p2 + xi3 * self.p3;
    }
}

fn main() {
    println!("{}",problems::shape_func::approx_func([1.0f64,1.0,1.0,1.0,1.0,1.0], 0.5, 0.5, 0.0));

    let p1 = new_point_2D(1.0, 1.0);
    let p2 = new_point_2D(3., 4.0);
    let p3 = new_point_2D(2.0, 1.0);

    let tri = Triangle2D {
        p1,p2,p3
    };

    let point_in_tri = tri.get_point_barycentric(1./3., 1./3., 1./3.);
    
    let inter_vec = interpolation_function_vector(array![0.5,0.5]);
    let inter_dev = interpolation_function_derivatives_vector(array![0.5,0.5]);
    let dom = domain_2D_matrix();
    let coordmat = tri.get_barycentric_matrix();

    println!("{}",coordmat);
    println!("{}",dom);
    println!("{}",inter_vec);

    let res = &coordmat.dot( &dom.dot( &inter_vec)) ;
    let res2 = &coordmat.dot( &dom.dot( &inter_dev)) ;

    let det = det2x2(res2); 

    println!("{}",res);
    println!("{},\n{}",res2, det);


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

