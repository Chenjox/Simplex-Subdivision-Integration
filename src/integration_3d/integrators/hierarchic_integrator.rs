use std::mem;

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
            // Simplex 1 (1 5 7 8)
            [1. , 0.5, 0.5, 0.5],
            [0.0, 0.5, 0. , 0. ],
            [0. , 0. , 0.5, 0. ],
            [0. , 0. , 0. , 0.5]
        ],
        array![
            // Simplex 2 (5 2 6 9)
            [0.5, 0., 0., 0.],
            [0.5, 1., 0.5, 0.5],
            [0., 0., 0.5, 0.],
            [0., 0., 0., 0.5]
        ],
        array![
            // Simplex 3 (7 6 3 10)
            [0.5, 0. , 0., 0. ],
            [0. , 0.5, 0., 0. ],
            [0.5, 0.5, 1., 0.5],
            [0. , 0. , 0., 0.5]
        ],
        array![
            // Simplex 4 (8 9 10 4)
            [0.5, 0.0, 0. , 0.],
            [0. , 0.5, 0. , 0.],
            [0.0, 0. , 0.5, 0.],
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
            //Simplex O,4 (10 16 17 19)
            [0., 0., 0., 0.5],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0.5, 0.5, 0., 0.],
            [0.5, 0., 0.5, 0.],
            [0., 0.5, 0.5, 0.5]
        ],
        array![
            //Simplex O,5 (15 16 8 19)
            [0., 0. , 0., 0.5],
            [0., 0., 0., 0.],
            [0.5, 0., 0.5, 0.],
            [0., 0.5, 0.5, 0.],
            [0., 0., 0., 0.],
            [0.5, 0.5, 0., 0.5]
        ],
        array![
            //Simplex O,6 (14 18 15 19)
            [0.,  0. , 0.,   0.5],
            [0.5, 0.5, 0.,   0.],
            [0.5, 0. , 0.5,  0.],
            [0.,  0. , 0.,   0.],
            [0.,  0. , 0.,   0.],
            [0.,  0.5, 0.5,  0.5]
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
            //Simplex O,8 (12 17 18 19)
            [0., 0., 0., 0.5],
            [0.5, 0., 0.5, 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0.5, 0.5, 0., 0.],
            [0., 0.5, 0.5, 0.5]
        ],
        array![
            // O 5 FIXME: Eventuell ersten und letzten Eintrag tauschen
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
            // O O,2 (13 2 14 19 12 18)
            [0.5, 0. , 0. , 0.5, 0. , 0. ],
            [0.5, 1. , 0.5, 0. , 0.5, 0.5],
            [0. , 0. , 0.5, 0. , 0. , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0. ],
            [0. , 0. , 0. , 0. , 0.5, 0. ],
            [0. , 0. , 0. , 0.5, 0. , 0.5],
        ],
        array![
            // O O,3 (7 3 8 19 14 15)
            [0.5, 0. , 0. , 0.5, 0. , 0. ],
            [0. , 0. , 0. , 0. , 0.5, 0. ],
            [0.5, 1. , 0.5, 0. , 0.5, 0.5],
            [0. , 0. , 0.5, 0. , 0. , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0. ],
            [0. , 0. , 0. , 0.5, 0. , 0.5],
        ],
        array![
            // O O,4 (9 4 10 19 8 16)
            [0.5, 0. , 0. , 0.5, 0. , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0. ],
            [0. , 0. , 0. , 0. , 0.5, 0. ],
            [0.5, 1. , 0.5, 0. , 0.5, 0.5],
            [0. , 0. , 0.5, 0. , 0. , 0. ],
            [0. , 0. , 0. , 0.5, 0. , 0.5],
        ],
        array![
            // O O,5 (11 5 12 19 10 17)
            [0.5, 0. , 0. , 0.5, 0. , 0. ],
            [0. , 0. , 0.5, 0. , 0. , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0. ],
            [0. , 0. , 0. , 0. , 0.5, 0.5],
            [0.5, 1. , 0.5, 0. , 0.5, 0. ],
            [0. , 0. , 0. , 0.5, 0. , 0.5],
        ],
        array![
            // O O,6 (19 15 16 17 18 6)
            [0.5, 0. , 0. , 0. , 0. , 0. ],
            [0. , 0. , 0. , 0. , 0.5, 0. ],
            [0. , 0.5, 0. , 0. , 0. , 0. ],
            [0. , 0. , 0.5, 0. , 0. , 0. ],
            [0. , 0. , 0. , 0.5, 0. , 0. ],
            [0.5, 0.5, 0.5, 0.5, 0.5, 1. ],
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
        //println!("{:?}",parent_vector);
        let subdivision_transformations = &subdivision_transformations();
        // Transformation ist ein Standardfall abhängig von dem ersten Kind
        
        let mut result = if is_tetrahedron_domain_number(parent_vector[0]) { array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ] } else {
            array![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]
        };
        for i in 0..parent_vector.len() - 1 {
            let current = parent_vector[i];
            // Die Zahlen gehen bis 4, 0 ist besonders.
            let current_transformation = &subdivision_transformations[(current - 1) as usize];
            //println!("{}\n{}",current,current_transformation);
            result = current_transformation.dot(&result);
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
                "T: Die Transformationsmatrix ist nicht der Dimension 4 x 4, sondern {} x {}",
                transformation.shape()[0],
                transformation.shape()[1]
            )
        }
        let result = self.base_integrator.integrate_over_domain(
            transformation,
            func,
            simplex,
            &mut IntegratorDummy::get(),
        );
        println!("{}", result);
        result
    }

    fn integrate_octahedron<T: Simplex3DFunction>(
        &self,
        transformation: &Array2<f64>,
        func: &Box<T>,
        simplex: &Simplex3D,
    ) -> f64 {
        if !(transformation.shape()[0] == 4 && transformation.shape()[1] == 6) {
            panic!(
                "O: Die Transformationsmatrix ist nicht der Dimension 4 x 6, sondern {} x {}",
                transformation.shape()[0],
                transformation.shape()[1]
            )
        }
        let octahedron_subdivisions = &octahedron_subdivisions();
        let mut result = 0.0;
        for i in 0..4 {
            let trans = &transformation.dot(&octahedron_subdivisions[i]);
            let temp_result = self.base_integrator.integrate_over_domain(
                &trans,
                func,
                simplex,
                &mut IntegratorDummy::get(),
            );
            result += temp_result;
            println!("{},{}", i,temp_result);
        }
        result
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
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

    /// MagicNumberMadness
    /// 20 is opening, 21 is closing
    pub fn new_cache_from_vec_tree(vec_tree: &Vec<u8>) -> Self {
        let arena = &mut Arena::new();
        
        // Zeiger
        let mut index = 0;
        // Stackw
        let mut stack = Vec::new();
        let mut root_node_id = None;
        loop {
            let element = vec_tree[index];
            match element {
                0 => { // Ist es Null, dann ist das der erste Knoten
                    let node_id = arena.new_node(NodeData::new(false, element));
                    root_node_id = Some(node_id);
                    stack.push(node_id);
                    // FIXME: hier fehlt sehr viel Error checking
                }
                1..=19 => {
                    // Wir stellen fest, das muss ein Knoten des Baums sein (Index 1 .. 19)
                    let node_id = arena.new_node(NodeData::new(false, element));

                    // Dieser muss einen Elternknoten haben, welcher auf dem Stack ist.
                    let parent : &NodeId = stack.last().unwrap();
                    parent.append(node_id, arena);

                    // ist der nächste 
                    if let Some(next_id) = vec_tree.get(index+1) {
                        if *next_id == 20_u8 {
                            stack.push(node_id);
                        }
                    } 
                    
                }
                20 => {}
                21 => {
                    stack.pop();
                }
                _ => {
                    panic!()
                }
            }
            
            if stack.is_empty() {
                break;
            }
            index +=1;
        }
        // Die Arena ist fertig, aber sie wird formal noch geteilt
        // Um Eigentümer davon zu werden, muss ich mir das Eigentum nehmen. (take ownership)
        let arena = mem::take(arena);
        match root_node_id {
            None => {
                panic!("No Root Node found!")
            }
            Some(root) => {

                println!("{:?}",root.debug_pretty_print(&arena));
                let arena = arena;
                return Self {
                    cached: true,
                    arena: arena,
                    root_node_id: root
                };
            }
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

                    // Jetzt wird das Integral des Blatts bestimmt.
                    let trans = Hierarchic3DIntegrator::<I>::get_transformation(&vec);
                    let child_transform = transformation.dot(&trans);

                    // Fallunterscheidung: Ist es ein Oktaeder oder ein Tetraeder?
                    println!("-- {:?}",vec);
                    let mut current_result = if tree[current_id].get().is_simplex_subdomain() {
                        self.integrate_tetrahedron(&child_transform, func, simplex)
                    } else {
                        self.integrate_octahedron(&child_transform, func, simplex)
                    };
                    //if tree[current_id].get().is_octahedral_subdomain() && current_result > 0.0 { println!("{},{:?}",current_result,vec); }

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
                    //println!("{},{}",result,current_result);
                    result += current_result;
                }

                //*tree[current_id].get_mut() = 42;
            }

            //println!("{}",result);
        } // Iteration ende

        return result;
    }
}
