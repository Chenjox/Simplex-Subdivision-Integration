use indextree::{Arena, NodeEdge, NodeId};
use ndarray::prelude::*;

use crate::integration_3d::{
    domain::{Simplex3D, Simplex3DFunction, Simplex3DIntegrator},
    IntegratorDummy,
};

pub struct Hierarchic3DIntegrator<I: Simplex3DIntegrator<IntegratorDummy>> {
    base_integrator: I,
    consolidated: bool,
    precision: f64,
}

fn subdivision_transformations() -> [Array2<f64>; 19] {
    [
        array![
            // Simplex 1
            [1., 0.5, 0.5, 0.5],
            [0.0, 0.5, 0., 0.],
            [0., 0., 0.5, 0.],
            [0., 0., 0., 0.5]
        ],
        array![
            // Simplex 2
            [0.5, 0., 0., 0.],
            [0.5, 1., 0.5, 0.5],
            [0., 0., 0.5, 0.],
            [0., 0., 0., 0.5]
        ],
        array![
            // Simplex 3
            [0.5, 0.5, 0., 0.],
            [0., 0., 0., 0.],
            [0.5, 0., 1., 0.5],
            [0., 0.5, 0., 0.5]
        ],
        array![
            // Simplex 4
            [0.5, 0.0, 0., 0.],
            [0., 0.5, 0., 0.],
            [0.0, 0., 0.5, 0.],
            [0.5, 0.5, 0.5, 1.]
        ],
        array![
            //Simplex O,1
            [0.5, 0.5, 0., 0.5],
            [0., 0.5, 0.5, 0.],
            [0.5, 0., 0.5, 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.5]
        ],
        array![
            //Simplex O,2
            [0.5, 0., 0.5, 0.5],
            [0., 0., 0., 0.],
            [0.5, 0.5, 0., 0.],
            [0., 0.5, 0.5, 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.5]
        ],
        array![
            //Simplex O,3
            [0.5, 0., 0.5, 0.5],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0.5, 0.5, 0., 0.],
            [0., 0.5, 0.5, 0.],
            [0., 0., 0., 0.5]
        ],
        array![
            //Simplex O,4
            [0., 0.5, 0., 0.5],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0.5, 0., 0.5, 0.],
            [0.5, 0.5, 0., 0.],
            [0., 0., 0.5, 0.5]
        ],
        array![
            //Simplex O,5
            [0., 0.5, 0., 0.5],
            [0., 0., 0., 0.],
            [0.5, 0., 0.5, 0.],
            [0., 0., 0.5, 0.],
            [0., 0.5, 0., 0.],
            [0.5, 0., 0., 0.5]
        ],
        array![
            //Simplex O,6
            [0., 0., 0., 0.5],
            [0.5, 0., 0.5, 0.],
            [0.5, 0.5, 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0.5, 0.5, 0.5]
        ],
        array![
            //Simplex O,7
            [0., 0.5, 0., 0.5],
            [0.5, 0.5, 0.5, 0.],
            [0.5, 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0.5, 0.],
            [0., 0., 0., 0.5]
        ],
        array![
            //Simplex O,8
            [0., 0., 0., 0.5],
            [0.5, 0., 0.5, 0.],
            [0., 0., 0., 0.],
            [0., 0.5, 0., 0.],
            [0.5, 0., 0., 0.],
            [0., 0.5, 0.5, 0.5]
        ],
        array![
            // O 5
            [0.5, 0., 0.5, 0.5, 0., 0.],
            [0.5, 0.5, 0., 0., 0.5, 0.],
            [0., 0.5, 0.5, 0., 0., 0.5],
            [0., 0., 0., 0.5, 0.5, 0.5],
        ],
        array![
            // O O,1
            [1., 0.5, 0.5, 0.5, 0.5, 0.5],
            [0., 0., 0., 0., 0.5, 0.],
            [0., 0.5, 0., 0., 0., 0.],
            [0., 0., 0.5, 0., 0., 0.],
            [0., 0., 0., 0.5, 0., 0.],
            [0., 0., 0., 0., 0., 0.5],
        ],
        array![
            // O O,2
            [0., 0., 0.5, 0., 0., 0.5],
            [1., 0.5, 0.5, 0.5, 0.5, 0.],
            [0., 0., 0., 0.5, 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0.5, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.5, 0.5],
        ],
        array![
            // O O,3
            [0., 0.5, 0., 0., 0., 0.5],
            [0., 0., 0., 0., 0.5, 0.],
            [1., 0.5, 0.5, 0.5, 0.5, 0.],
            [0., 0., 0.5, 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0.5, 0., 0.5],
        ],
        array![
            // O O,4
            [0., 0., 0.5, 0., 0.5, 0.5],
            [0., 0., 0., 0., 0., 0.],
            [0., 0.5, 0., 0., 0., 0.],
            [1., 0.5, 0.5, 0.5, 0., 0.],
            [0., 0., 0., 0.5, 0.5, 0.],
            [0., 0., 0., 0., 0., 0.5],
        ],
        array![
            // O O,5
            [0., 0., 0.5, 0., 0., 0.5],
            [0., 0., 0., 0.5, 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0.5, 0., 0., 0.5, 0.],
            [1., 0.5, 0.5, 0.5, 0., 0.],
            [0., 0., 0., 0., 0.5, 0.5],
        ],
        array![
            // O O,6
            [0., 0., 0.5, 0., 0., 0.5],
            [0., 0., 0., 0., 0.5, 0.],
            [0., 0.5, 0., 0., 0., 0.],
            [0., 0., 0., 0.5, 0., 0.],
            [0., 0., 0.5, 0., 0., 0.],
            [1., 0.5, 0., 0.5, 0.5, 0.5],
        ],
    ]
}

fn octahedron_subdivisions() -> [Array2<f64>; 4] {
    [
        array![
            // O 1
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.],
        ],
        array![
            // O 2
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.],
        ],
        array![
            // O 3
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 1., 0.],
        ],
        array![
            // O 4
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [0., 1., 0., 0.],
        ],
    ]
}

impl<I: Simplex3DIntegrator<IntegratorDummy>> Hierarchic3DIntegrator<I> {
    pub fn new(base_integrator: I, consolidated: bool, precision: f64) -> Self {
        Self {
            base_integrator,
            consolidated,
            precision,
        }
    }

    fn get_transformation(parent_vector: &Vec<u8>) -> Array2<f64> {
        // Der höchste Index ist 18
        let subdivision_transformations = &subdivision_transformations();
        let mut result = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        for i in 0..parent_vector.len() - 1 {
            let current = parent_vector[i];
            // Die Zahlen gehen bis 4, 0 ist besonders.
            let current_transformation = &subdivision_transformations[(current - 1) as usize];
            result = result.dot(current_transformation);
        }
        return result;
    }

    fn integrate_tetrahedron<T: Simplex3DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex3D,
    ) -> f64 {
        if !(transformation.shape()[0] == 4 && transformation.shape()[1] == 4) {
            panic!(
                "Die Transformationsmatrix ist nicht der Dimension 4 x 4, sondern {} x {}",
                transformation.shape()[0],
                transformation.shape()[1]
            )
        }
        self.base_integrator.integrate_over_domain(
            transformation,
            func,
            simplex,
            &mut IntegratorDummy::get(),
        )
    }

    fn integrate_octahedron<T: Simplex3DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex3D,
    ) -> f64 {
        if !(transformation.shape()[0] == 4 && transformation.shape()[1] == 6) {
            panic!(
                "Die Transformationsmatrix ist nicht der Dimension 4 x 6, sondern {} x {}",
                transformation.shape()[0],
                transformation.shape()[1]
            )
        }
        let octahedron_subdivisions = &octahedron_subdivisions();
        let mut result = 0.0;
        for i in 0..4 {
            let trans = &octahedron_subdivisions[i].dot(transformation);
            result += self.base_integrator.integrate_over_domain(
                &trans,
                func,
                simplex,
                &mut IntegratorDummy::get(),
            )
        }
        result
    }
}

struct NodeData {
    checked: bool,
    number: u8,
}

/// See Theory PDF under Arbitrary decisions.
/// All Magic numbers are checked here and only here.
fn is_tetrahedron_domain_number(num: u8) -> bool {
    if num > 19 {
        panic!("Illegal Magic number for subdomain! {}", num)
    }
    num < 13
}

/// see [`is_tetrahedron_domain_number()`]
fn is_octahedron_domain_number(num: u8) -> bool {
    !is_tetrahedron_domain_number(num)
}

impl NodeData {
    fn new(checked: bool, number: u8) -> Self {
        return Self { checked, number };
    }

    /// see [`is_tetrahedron_domain_number()`]
    fn is_simplex_subdomain(&self) -> bool {
        is_tetrahedron_domain_number(self.number)
    }

    /// see [`is_octahedron_domain_number()`]
    fn is_octahedral_subdomain(&self) -> bool {
        is_octahedron_domain_number(self.number)
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

                    // Fallunterscheidung: Ist es ein Oktaeder oder ein Tetraeder?
                    let mut current_result = if tree[current_id].get().is_simplex_subdomain() {
                        self.integrate_tetrahedron(&child_transform, func, simplex)
                    } else {
                        self.integrate_octahedron(&child_transform, func, simplex)
                    };

                    // Wenn das Blatt noch nicht überprüft worden ist und noch nicht consolidiert ist.
                    if !tree[current_id].get().checked && !self.consolidated {
                        // Dann wird eine Verfeinerungsstufe mehr eingebaut.
                        //todo!("AB hier wirds kritisch, verfeinerung muss zwischen oktaeder und tetraeder unterscheiden!");

                        let mut child_result = 0.;
                        // Wenn die kleinste subdomain ein Simplex ist:
                        if tree[current_id].get().is_simplex_subdomain() {
                            {
                                // Dann gibt es 4 (Index 1 bis 4) Kindtetraeder
                                for i in 0..4 {
                                    let i_1 = i + 1;
                                    let mut child_vec = vec.clone();
                                    // die temporäre transformationshierachie
                                    child_vec.insert(0, i_1 as u8);
                                    let child_trans =
                                        Hierarchic3DIntegrator::<I>::get_transformation(&child_vec);
                                    let child_transformation = transformation.dot(&child_trans);
                                    child_result += self.integrate_tetrahedron(
                                        &child_transformation,
                                        func,
                                        simplex,
                                    );
                                }
                                // und 1 Kindoktaeder (Index 13)
                                {
                                    let i_1 = 13;
                                    let mut child_vec = vec.clone();

                                    child_vec.insert(0, i_1 as u8);
                                    let child_trans =
                                        Hierarchic3DIntegrator::<I>::get_transformation(&child_vec);
                                    let child_transformation = transformation.dot(&child_trans);
                                    child_result += self.integrate_octahedron(
                                        &child_transformation,
                                        func,
                                        simplex,
                                    );
                                }
                            }
                        } else {
                            // Wenn ein Oktaeder Verfeinert werden muss:
                            {
                                // Dann gibt es 8 (Index 5 bis 12) Kindtetraeder
                                for i in 0..8 {
                                    let i_1 = i + 5;
                                    let mut child_vec = vec.clone();
                                    // die temporäre transformationshierachie
                                    child_vec.insert(0, i_1 as u8);
                                    let child_trans =
                                        Hierarchic3DIntegrator::<I>::get_transformation(&child_vec);
                                    let child_transformation = transformation.dot(&child_trans);
                                    child_result += self.integrate_tetrahedron(
                                        &child_transformation,
                                        func,
                                        simplex,
                                    );
                                }
                                // und 6 Kindoktaeder (Index 14 bis 19)
                                for i in 0..6 {
                                    let i_1 = i + 14;
                                    let mut child_vec = vec.clone();
                                    // die temporäre transformationshierachie
                                    child_vec.insert(0, i_1 as u8);
                                    let child_trans =
                                        Hierarchic3DIntegrator::<I>::get_transformation(&child_vec);
                                    let child_transformation = transformation.dot(&child_trans);
                                    child_result += self.integrate_octahedron(
                                        &child_transformation,
                                        func,
                                        simplex,
                                    );
                                }
                            }
                        }

                        // Wenn die Verfeinerung "genauer" ist, dann wird der Baum angepasst.
                        // Und das Element wurde überprüft.
                        tree[current_id].get_mut().checked = true;
                        if (current_result - child_result).abs() > precision_threshold {
                            if tree[current_id].get().is_simplex_subdomain() {
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
                                current_id.append(
                                    tree.new_node(NodeData {
                                        checked: false,
                                        number: 13,
                                    }),
                                    tree,
                                )
                            } else {
                                for i in 0..8 {
                                    let i = i + 5;
                                    current_id.append(
                                        tree.new_node(NodeData {
                                            checked: false,
                                            number: i,
                                        }),
                                        tree,
                                    );
                                }
                                for i in 0..6 {
                                    let i = i + 14;
                                    current_id.append(
                                        tree.new_node(NodeData {
                                            checked: false,
                                            number: i,
                                        }),
                                        tree,
                                    );
                                }
                            }
                            // dem Element fügen wir die Kinder hinzu

                            // Der Baum hat sich geändert!
                            state_changed = true;
                            // Das Resultat ist das genauere resultat
                            current_result = child_result;
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
