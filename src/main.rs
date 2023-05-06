use indextree::{Arena, NodeEdge, NodeId};
use ndarray::prelude::*;
use std::fmt::write;
use std::fs::File;
use std::io::Write;

use crate::{
    integration_2d::{functions::Function2DHistory, integrators::quadrilaterial_integrator::*, *},
    problems::PhaseField2DFunction,
};

mod integration_2d;
mod problems;

struct Hierarchic2DIntegration<I: Simplex2DIntegrator<IntegratorDummy>> {
    base_integrator: I,
    consolidated: bool,
}

impl<I: Simplex2DIntegrator<IntegratorDummy>> Hierarchic2DIntegration<I> {
    fn subdivision_transformations() -> [Array2<f64>; 4] {
        return [
            array![
                // S1
                [1., 0.5, 0.5],
                [0., 0.5, 0.],
                [0., 0., 0.5]
            ],
            array![
                // S2
                [0., 0.0, 0.5],
                [1., 0.5, 0.5],
                [0., 0.5, 0.0]
            ],
            array![
                // S3
                [0., 0.5, 0.0],
                [0., 0.0, 0.5],
                [1., 0.5, 0.5]
            ],
            array![
                // S4
                [0.5, 0.5, 0.0],
                [0., 0.5, 0.5],
                [0.5, 0.0, 0.5]
            ],
        ];
    }

    fn get_transformation(parent_vector: &Vec<u8>) -> Array2<f64> {
        let transformations = Hierarchic2DIntegration::<I>::subdivision_transformations();
        let mut result = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..parent_vector.len() - 1 {
            let current = parent_vector[i];
            let current_transformation = &transformations[(current - 1) as usize];
            result = current_transformation.dot(&result);
        }
        return result;
    }
}

struct NodeData {
    checked: bool,
    number: u8,
}

impl NodeData {
    fn new(checked: bool, number: u8) -> Self {
        return Self { checked, number };
    }
}

struct Hierarchic2DIntegratorData {
    cached: bool,
    root_node_id: NodeId,
    arena: Arena<NodeData>,
}

impl Hierarchic2DIntegratorData {
    fn new_cache() -> Self {
        let mut arena = Arena::new();
        let root = arena.new_node(NodeData::new(false, 0));
        Self {
            cached: false,
            arena: arena,
            root_node_id: root,
        }
    }
}

impl<I: Simplex2DIntegrator<IntegratorDummy>> Simplex2DIntegrator<Hierarchic2DIntegratorData>
    for Hierarchic2DIntegration<I>
{
    fn integrate_over_domain<T: Simplex2DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex2D,
        cached_data: &mut Hierarchic2DIntegratorData,
    ) -> f64 {
        // Sollte cached_data noch nicht initialisiert worden sein, dann wirds zeit
        // Danach ist der Cache grundsätzlich gültig.
        if !cached_data.cached {
            cached_data.arena = Arena::new();
            cached_data.root_node_id = cached_data.arena.new_node(NodeData::new(false, 0));
            cached_data.cached = true;
        }
        // It all begins with a tree!
        let tree = &mut cached_data.arena;
        let root_node_id = cached_data.root_node_id;

        let mut state_changed = true;
        let precision_threshold = 0.00001;
        let mut result = 0.;

        while state_changed {
            // Grundsätzlich wird sich der Baum nicht ändern
            state_changed = false;
            // Das Integral wird von vorn Integriert.
            result = 0.;
            // Alle Knoten DFS durchgehen
            let mut next_edge = Some(NodeEdge::Start(root_node_id));
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
                    vec.push(tree[current_id].get().number);

                    let mut par = current_id;
                    while let Some(parent) = tree[par].parent() {
                        vec.push(tree[parent].get().number);
                        par = parent;
                    }
                    // Alle Elternknoten wurden ermittelt.
                    // `vec` soll nicht mehr bearbeitet werden
                    let vec = vec;
                    //println!("{:?}", vec);

                    // Jetzt wird das Integral des Blatts bestimmt.
                    let trans = Hierarchic2DIntegration::<I>::get_transformation(&vec);
                    let child_transform = transformation.dot(&trans);
                    let mut current_result = self.base_integrator.integrate_over_domain(
                        &child_transform,
                        func,
                        simplex,
                        &mut IntegratorDummy::get(),
                    );

                    // Wenn das Blatt noch nicht überprüft worden ist und noch nicht consolidiert ist.
                    if !tree[current_id].get().checked && !self.consolidated {
                        // Dann wird eine Verfeinerungsstufe mehr eingebaut.
                        let mut child_result = 0.;
                        for i in 0..4 {
                            let i_1 = i + 1;
                            let mut child_vec = vec.clone();
                            // die temporäre transformationshierachie
                            child_vec.insert(0, i_1 as u8);
                            let child_trans =
                                Hierarchic2DIntegration::<I>::get_transformation(&vec);
                            let child_transformation = transformation.dot(&child_trans);
                            child_result += self.base_integrator.integrate_over_domain(
                                &child_transformation,
                                func,
                                simplex,
                                &mut IntegratorDummy::get(),
                            );
                        }
                        // Wenn die Verfeinerung "genauer" ist, dann wird der Baum angepasst.
                        if (current_result - child_result).abs() > precision_threshold {
                            // Dieses Element wurde geprüft
                            tree[current_id].get_mut().checked = true;
                            // dem Element fügen wir die Kinder hinzu
                            for i in 0..4 {
                                let i = i + 1;
                                current_id.append(
                                    tree.new_node(NodeData {
                                        checked: false,
                                        number: i,
                                    }),
                                    tree,
                                );
                            }
                            // Der Baum hat sich geändert!
                            state_changed = true;
                            // Das Resultat ist das genauere resultat
                            current_result = child_result;
                        } else {
                            tree[current_id].get_mut().checked = true;
                        }
                    }
                    result += current_result;
                }

                //*tree[current_id].get_mut() = 42;
            }

            //println!("{}",result);
        } // Iteration ende

        return result;
    }
}

fn main() {
    // ASSERTION: The Simplex is always rightly oriented.
    let sim = Simplex2D::new_from_points(
        &array![1., 1.],
        &array![1.5, 1. + (3.0f64).sqrt() / 2.],
        &array![2., 1.],
    );
    let inte = Quadrilateral2DIntegrator::new(1);
    let inte = Hierarchic2DIntegration {
        base_integrator: inte,
        consolidated: false,
    };

    let func = Box::new(Function2DHistory::new(PhaseField2DFunction {
        weights: [2.0, 2.0, 2.0, 2.0, -0., -0.],
    }));

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
    let result = inte.integrate_simplex(&func, &sim, &mut cache);
    println!("{}", result);

    func.delete_history();

    let result = inte.integrate_simplex(&func, &sim, &mut cache);

    let hist = func.get_history();

    println!("{},{}",result, hist.len());

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
