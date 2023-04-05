use indextree::Arena;
use ndarray::prelude::*;

mod problems;
mod domain;

type Point2D = Array2<f64>;
type Point3D = Array3<f64>;

struct Triangle {
    p1: Point2D,
    p2: Point2D,
    p3: Point2D
}

impl Triangle {
    fn get_point_barycentric(self, xi1: f64, xi2: f64, xi3: f64) -> Point2D {
        if xi1 + xi2 + xi3 > 1.0 {panic!("Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",xi1,xi2,xi3,xi1+xi2+xi3);};
        return xi1 * self.p1 + xi2 * self.p2 + xi3 * self.p3;
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
}

