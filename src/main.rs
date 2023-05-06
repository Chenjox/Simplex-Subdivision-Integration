use indextree::{Arena, NodeEdge, NodeId};
use ndarray::prelude::*;

use crate::{integration_2d::{*, integrators::quadrilaterial_integrator::*, functions::Function2DHistory}, problems::PhaseField2DFunction};

mod integration_2d;
mod problems;


struct Hierarchic2DIntegration<I: Simplex2DIntegrator> {
    base_integrator: I,
}

impl<I: Simplex2DIntegrator> Hierarchic2DIntegration<I> {
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
            result = result.dot(current_transformation)
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

impl<I: Simplex2DIntegrator> Simplex2DIntegrator for Hierarchic2DIntegration<I> {
    fn integrate_over_domain<T: Simplex2DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex2D,
    ) -> f64 {
        // It all begins with a tree!
        let tree = &mut Arena::new();
        let root_node_id = tree.new_node(NodeData::new(false, 0));

        let mut state_changed = true;
        let precision_threshold = 0.001;
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
                    let mut current_result =
                        self.base_integrator
                            .integrate_over_domain(&child_transform, func, simplex);

                    // Wenn das Blatt noch nicht überprüft worden ist
                    if !tree[current_id].get().checked {
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
                            );
                        }
                        // Wenn die Verfeinerung "genauer" ist, dann wird der Baum angepasst.
                        if (current_result - child_result).abs() > precision_threshold { 
                            tree[current_id].get_mut().checked = true;
                            for i in 1..5 {
                                current_id.append(
                                    tree.new_node(NodeData {
                                        checked: false,
                                        number: i,
                                    }),
                                    tree,
                                );
                            }
                            state_changed = true;
                            current_result = child_result;
                        } else {
                            tree[current_id].get_mut().checked = true;
                        }
                    }
                    result += current_result;
                    
                }

                //*tree[current_id].get_mut() = 42;
            }
            
            println!("{}",result);
        }// Iteration ende
        
        return result;
    }
}

fn main() {
    // ASSERTION: The Simplex is always rightly oriented.
    let sim = Simplex2D::new_from_points(&array![1., 1.], &array![1., 2.], &array![2., 1.]);
    let inte = Quadrilateral2DIntegrator::new(2);
    let inte = Hierarchic2DIntegration {
        base_integrator: inte,
    };

    let func = Box::new(
        Function2DHistory::new(
            PhaseField2DFunction {
                weights: [1.0,1.0,-1.0,1.0,-1.0,1.0]
            }
        )
    );

    let result = inte.integrate(&func, &sim);
    println!("{}", result);

    let hist = func.get_history();

    println!("{}",hist.len());
    /*
    for el in hist {
        //println!("{},{}",el, el.fold(0., |f1, f2| f1 + f2));
        println!("\\draw[fill,red] (barycentric cs:ca={:.3},cb={:.3},cc={:.3}) coordinate (cb1) circle (2pt);",el[0],el[1],el[2]);
    } */

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
