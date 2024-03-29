use indextree::{Arena, NodeEdge, NodeId};
use ndarray::prelude::*;

use crate::common::IntegratorDummy;
use crate::integration_2d::domain::{
    Simplex2D, Simplex2DFunction, Simplex2DIntegrator, Simplex2DResultType,
};

pub struct Hierarchic2DIntegrator<I: Simplex2DIntegrator<IntegratorDummy>> {
    base_integrator: I,
    consolidated: bool,
    precision: f64,
}

impl<I: Simplex2DIntegrator<IntegratorDummy>> Hierarchic2DIntegrator<I> {
    pub fn new(base_integrator: I, consolidated: bool, precision: f64) -> Self {
        Self {
            base_integrator,
            consolidated,
            precision,
        }
    }
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
        let transformations = Hierarchic2DIntegrator::<I>::subdivision_transformations();
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

pub struct Hierarchic2DIntegratorData {
    cached: bool,
    root_node_id: NodeId,
    arena: Arena<NodeData>,
}

impl Hierarchic2DIntegratorData {
    pub fn new_cache() -> Self {
        let mut arena = Arena::new();
        let root = arena.new_node(NodeData::new(false, 0));
        Self {
            cached: false,
            arena: arena,
            root_node_id: root,
        }
    }

    pub fn make_leafs_unchecked(&mut self) {
        let tree = &mut self.arena;
        let root_node_id = self.root_node_id;

        let mut next_edge = Some(NodeEdge::Start(root_node_id));
        while let Some(current_edge) = next_edge {
            next_edge = current_edge.next_traverse(&tree);
            let current_id = match current_edge {
                NodeEdge::Start(_) => continue,
                NodeEdge::End(id) => id,
            };

            // Wenn der mometane Knoten ein Blattknoten ist, dann....
            if tree[current_id].first_child().is_none() {
                tree[current_id].get_mut().checked = false;
            }
        }
    }

    pub fn tree_size(&self) -> usize {
        return self.arena.count();
    }
}

impl<I: Simplex2DIntegrator<IntegratorDummy>> Simplex2DIntegrator<Hierarchic2DIntegratorData>
    for Hierarchic2DIntegrator<I>
{
    fn dupe(&self) -> Self {
        Self {
            base_integrator: self.base_integrator.dupe(),
            precision: self.precision,
            consolidated: self.consolidated,
        }
    }

    fn integrate_over_domain<T: Simplex2DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex2D,
        cached_data: &mut Hierarchic2DIntegratorData,
    ) -> T::Return {
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
        let mut result = func.additive_neutral_element();

        while state_changed {
            // Grundsätzlich wird sich der Baum nicht ändern
            state_changed = false;
            // Das Integral wird von vorn Integriert.
            result = func.additive_neutral_element();
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
                    // Vec fängt beim Kind an, hört bei root auf
                    let vec = vec;
                    //println!("{:?}", vec);

                    // Jetzt wird das Integral des Blatts bestimmt.
                    let trans = Hierarchic2DIntegrator::<I>::get_transformation(&vec);
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
                        let mut child_result = func.additive_neutral_element();
                        for i in 0..4 {
                            let i_1 = i + 1;
                            let mut child_vec = vec.clone();
                            // die temporäre transformationshierachie
                            child_vec.insert(0, i_1 as u8);
                            let child_trans =
                                Hierarchic2DIntegrator::<I>::get_transformation(&child_vec);
                            let child_transformation = transformation.dot(&child_trans);
                            child_result.add_assign(&self.base_integrator.integrate_over_domain(
                                &child_transformation,
                                func,
                                simplex,
                                &mut IntegratorDummy::get(),
                            ));
                        }
                        // Wenn die Verfeinerung "genauer" ist, dann wird der Baum angepasst.
                        if current_result.distance(&child_result).abs() > precision_threshold {
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
                    result.add_assign(&current_result);
                }

                //*tree[current_id].get_mut() = 42;
            }

            //println!("{}",result);
        } // Iteration ende

        return result;
    }
}

#[cfg(test)]
mod tests {
    use crate::common::IntegratorDummy;
    use crate::integration_2d::integrators::{
        Hierarchic2DIntegrator, Hierarchic2DIntegratorData, Quadrilateral2DIntegrator,
    };
    use crate::integrator_tests;

    integrator_tests! {
        quadrilaterial1: Hierarchic2DIntegrator<Quadrilateral2DIntegrator>: Hierarchic2DIntegrator::new(Quadrilateral2DIntegrator::new(1),false,1e-2), Hierarchic2DIntegratorData: Hierarchic2DIntegratorData::new_cache(),
    }
}
