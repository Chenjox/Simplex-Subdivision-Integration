use indextree::{Arena, NodeEdge, NodeId};
use ndarray::prelude::*;
use std::fmt::write;
use std::fs::File;
use std::io::Write;

use crate::{
    integration_2d::{functions::{Function2DHistory, Constant2DFunction}, integrators::{quadrilaterial_integrator::*, Hierarchic2DIntegratorData, Hierarchic2DIntegrator}, *},
    problems::PhaseField2DFunction,
};

mod integration_2d;
mod problems;

fn main() {
    // ASSERTION: The Simplex is always rightly oriented.
    let sim = Simplex2D::new_from_points(
        &array![1., 1.],
        &array![1.5, 1. + (3.0f64).sqrt() / 2.],
        &array![2., 1.],
    );
    let inte1 = Quadrilateral2DIntegrator::new(1);
    let inte2 = Hierarchic2DIntegrator::new(inte1, false, 1e-10);
    let inte1 = Quadrilateral2DIntegrator::new(1);

    let func = Box::new(Function2DHistory::new(PhaseField2DFunction {
        weights: [2.0, 2.0, 2.0, 2.0, -0., -0.],
    }));
    //let func = Box::new(Function2DHistory::new(Constant2DFunction));

    let mut cache = Hierarchic2DIntegratorData::new_cache();

    /*
        cache.cached = true;
        let mut inte = inte;
        {
            let arena = &mut cache.arena;
            let root = &cache.root_node_id;
            inte.consolidated = true;

            let mut node_id = None;
            for i in 0..4 {
                let i = i + 1;
                node_id = Some(arena.new_node(NodeData {
                    checked: false,
                    number: i,
                }));
                root.append(
                    node_id.unwrap(),
                    arena,
                );
            }

            let child_node_id = node_id.unwrap();
            for i in 0..4 {
                let i = i + 1;
                node_id = Some(arena.new_node(NodeData {
                    checked: false,
                    number: i,
                }));
                child_node_id.append(
                    node_id.unwrap(),
                    arena,
                );
            }
        }
        let inte = inte;
    */
    let result1 = inte1.integrate_simplex(&func, &sim, &mut IntegratorDummy::get());
    let result2 = inte2.integrate_simplex(&func, &sim, &mut cache);
    println!("{},{}", result1, result2);

    func.delete_history();

    let result = inte2.integrate_simplex(&func, &sim, &mut cache);

    let hist = func.get_history();

    println!("{},{}", result, hist.len());

    //for el in &hist {
    //    //println!("{},{}",el, el.fold(0., |f1, f2| f1 + f2));
    //    println!("\\draw[fill,red] (barycentric cs:ca={:.3},cb={:.3},cc={:.3}) coordinate (cb1) circle (2pt);",el[0],el[1],el[2]);
    //}

    let mut f = File::create("output.csv").expect("Unable to create file");
    let points = sim.get_points();
    for i in 0..3 {
        writeln!(f, "{} {}", points[[0, i]], points[[1, i]]).expect("Unable to write!");
    }
    for i in &hist {
        let p = points.dot(i);
        writeln!(f, "{} {}", p[0], p[1]).expect("Unable to write!");
    }

    /*
    let tree = &mut Arena::new();

    let a = tree.new_node(0);
    let b = tree.new_node(1);
    let c = tree.new_node(2);

    a.append(b, tree);
    a.append(c, tree);
    b.append(tree.new_node(4), tree);
    b.append(tree.new_node(5), tree);

    // Traversiere alle Knoten im Baum DFS
    let mut next_edge = Some(NodeEdge::Start(a));
    while let Some(current_edge) = next_edge {
        next_edge = current_edge.next_traverse(&tree);
        let current_id = match current_edge {
            NodeEdge::Start(_) => continue,
            NodeEdge::End(id) => id,
        };

        // Wenn der mometane Knoten ein Blattknoten ist, dann....
        if tree[current_id].first_child().is_none() {
            // Ermittle alle Vorfahren
            let mut vec = Vec::new();
            vec.push(tree[current_id].get());

            let mut par = current_id;
            while let Some(parent) = tree[par].parent() {
                vec.push(tree[parent].get());
                par = parent;
            }
            println!("{:?}", vec);

            // Wenn _kriterium erreicht_ dann
            if tree[current_id].get() == &5 {
                current_id.append(tree.new_node(6), tree);
                current_id.append(tree.new_node(6), tree);
            }
        }

        //todo!("it is possible to modify `arena[current_id]` here!");
        // *tree[current_id].get_mut() = 42;
    }

    //for node in a.descendants(tree) {
    //    println!("{}",tree[node].get())
    //}
    */
}
