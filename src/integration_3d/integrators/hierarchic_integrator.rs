use indextree::{Arena, NodeEdge, NodeId};
use ndarray::prelude::*;

use crate::integration_3d::{domain::{Simplex3D, Simplex3DFunction, Simplex3DIntegrator}, IntegratorDummy};

pub struct Hierarchic3DIntegrator<I: Simplex3DIntegrator<IntegratorDummy>> {
    base_integrator: I,
    consolidated: bool,
    precision: f64,
}

impl<I: Simplex3DIntegrator<IntegratorDummy>> Hierarchic3DIntegrator<I> {
    pub fn new(base_integrator: I, consolidated: bool, precision: f64) -> Self {
        Self {
            base_integrator,
            consolidated,
            precision,
        }
    }
    fn subdivision_transformations() -> [Array2<f64>; 4] {
        todo!("Implementent Subdivision transformations")
    }

    fn get_transformation(parent_vector: &Vec<u8>) -> Array2<f64> {
        todo!("Implement transformations!");
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

    /// Watch out! Magic Numbers!
    /// Whether it is a tetrahedral or octahedral domain.
    /// 
    fn is_simplex_subdomain(&self) -> bool {
        if self.number > 19 {
            panic!("Illegal Magic number! {}",self.number)
        }
        self.number > 0 && self.number < 13
    }

    /// Watch out! Magic Numbers!
    /// First will be the octahedrons
    fn is_octahedral_subdomain(&self) -> bool {
        !self.is_octahedral_subdomain()
    }
}

pub struct Hierarchic3DIntegratorData {
    cached: bool,
    root_node_id: NodeId,
    arena: Arena<NodeData>,
}

impl Hierarchic3DIntegratorData {
    pub fn new_cache() -> Self {
        let mut arena = Arena::new();
        let root = arena.new_node(NodeData::new(false, 0));
        Self {
            cached: false,
            arena: arena,
            root_node_id: root,
        }
    }
}

impl<I: Simplex3DIntegrator<IntegratorDummy>> Simplex3DIntegrator<Hierarchic3DIntegratorData>
    for Hierarchic3DIntegrator<I>
{
    fn integrate_over_domain<T: Simplex3DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex3D,
        cached_data: &mut Hierarchic3DIntegratorData,
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
        let precision_threshold = self.precision;
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
                    let trans = Hierarchic3DIntegrator::<I>::get_transformation(&vec);
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
                                Hierarchic3DIntegrator::<I>::get_transformation(&child_vec);
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
