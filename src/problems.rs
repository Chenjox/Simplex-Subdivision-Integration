use crate::{domain::Simplex2DFunction, integration_2d::domain::ResultTypeWrapper};

use self::{problem_definition::phase_field::phase_field_func, shape_func::approx_func};

const TOLERANCE: f64 = 1e-10;

pub mod shape_func {
    use super::TOLERANCE;

    /// Shape Function for the 1 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN1(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        2.0 * xi1 * (xi1 - 0.5)
    }
    /// Shape Function for the 2 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN2(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        2.0 * xi2 * (xi2 - 0.5)
    }
    /// Shape Function for the 3 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN3(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        2.0 * xi3 * (xi3 - 0.5)
    }
    /// Shape Function for the 4 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN4(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        4.0 * xi1 * xi3
    }
    /// Shape Function for the 5 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN5(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        4.0 * xi1 * xi2
    }
    /// Shape Function for the 6 Node of a 6-Node Triangular Element
    /// See theory.tex for clarificatiom
    pub fn BarN6(xi1: f64, xi2: f64, xi3: f64) -> f64 {
        if (xi1 + xi2 + xi3 - 1.0).abs() >= TOLERANCE {
            panic!(
                "Illegal Barycentric Coordinates, {},{},{} = {} > 1.0!",
                xi1,
                xi2,
                xi3,
                xi1 + xi2 + xi3
            );
        };
        4.0 * xi2 * xi3
    }

    pub fn approx_func(weights: [f64; 6], xi1: f64, xi2: f64, xi3: f64) -> f64 {
        weights[0] * BarN1(xi1, xi2, xi3)
            + weights[1] * BarN2(xi1, xi2, xi3)
            + weights[2] * BarN3(xi1, xi2, xi3)
            + weights[3] * BarN4(xi1, xi2, xi3)
            + weights[4] * BarN5(xi1, xi2, xi3)
            + weights[5] * BarN6(xi1, xi2, xi3)
    }
}

pub mod problem_definition {
    use ndarray::{array, Array1};

    use crate::integration_3d::Simplex3DFunction;

    use self::{
        phase_field::{phase_field_func, varsigma_func_diff1, varsigma_func_diff2},
        shape_func_3d::{ansatz_function, dyadic_product_component},
    };

    pub mod phase_field {

        // exponent of phase field
        pub fn varsigma_func(f_base: f64, kreg: f64) -> f64 {
            f_base / (f_base.powi(2) + kreg).powf(0.25)
        }

        pub fn varsigma_func_diff1(f_base: f64, kreg: f64) -> f64 {
            (1. - f_base.powi(2) / (2. * (f_base.powi(2) + kreg))) * 1.
                / (f_base.powi(2) + kreg).powf(0.25)
        }

        pub fn varsigma_func_diff2(f_base: f64, kreg: f64) -> f64 {
            ((5. * f_base.powi(3)) / (f_base.powi(2) + kreg) - 6. * f_base) * 1.
                / (4. * (f_base.powi(2) + kreg).powf(1.25))
        }

        pub fn varsigma_func_diff3(f_base: f64, kreg: f64) -> f64 {
            ((3. * (-4. * kreg.powi(2) + 12. * kreg * f_base.powi(2) + f_base.powi(4)))
                / (2. * (f_base.powi(2) + kreg).powi(2)))
                * 1.
                / (4. * (f_base.powi(2) + kreg).powf(1.25))
        }

        pub fn phase_field_func(f_base: f64, kreg: f64, l: f64) -> f64 {
            (-(varsigma_func(f_base, kreg)).abs() * 1.0 / l).exp()
        }

        //pub fn phase_field_func_diff2(f_base: f64, kreg: f64, l: f64, row_index: usize, column_index: usize) -> f64 {
        //    phase_field_func(f_base, kreg, l) * ( varsigma_func_diff1(f_base, kreg).powi(2) - varsigma_func_diff2(f_base, kreg) )
        //}
    }

    mod shape_func_2d {
        use ndarray::Array1;

        // Baryzentrische Lagrange Funktion [Index 0 - 2]
        fn lagrange_1_function(barycentric_coordinates: &Array1<f64>, index: usize) -> f64 {
            if index > 2 {
                panic!("Illegal Index given");
            };
            return barycentric_coordinates[index];
        }
        // Knoten 1 - 3 [Index 0 - 2]
        fn lagrange_2_order_nodal(barycentric_coordinates: &Array1<f64>, index: usize) -> f64 {
            if index > 2 {
                panic!("Illegal Index given");
            }
            return lagrange_1_function(barycentric_coordinates, index)
                * (2. * lagrange_1_function(barycentric_coordinates, index) - 1.);
        }

        // Knoten 4 - 6 [Index 3 - 5]
        fn lagrange_2_order_edge(barycentric_coordinates: &Array1<f64>, index: usize) -> f64 {
            if index < 3 {
                panic!("Illegal Index given");
            }
            if index < 6 {
                // 4,5,6
                let index = index - 3;
                return 4.
                    * lagrange_1_function(barycentric_coordinates, index)
                    * lagrange_1_function(barycentric_coordinates, 2 - index);
            } else {
                panic!("Illegal Index given")
            }
        }

        pub fn shape_function_from_index(
            barycentric_coordinates: &Array1<f64>,
            index: usize,
        ) -> f64 {
            if index > 5 {
                panic!("Illegal Index encountered")
            }
            if index < 3 {
                return lagrange_2_order_nodal(barycentric_coordinates, index);
            } else {
                return lagrange_2_order_edge(barycentric_coordinates, index);
            }
        }

        pub fn all_shape_functions(barycentric_coordinates: &Array1<f64>) -> Array1<f64> {
            let mut res = Array1::<f64>::zeros([6]);
            for i in 0..6 {
                res[i] = shape_function_from_index(barycentric_coordinates, i);
            }
            return res;
        }

        // [N \otimes N]_{ij}
        pub fn dyadic_product_component(
            barycentric_coordinates: &Array1<f64>,
            row_index: usize,
            column_index: usize,
        ) -> f64 {
            return shape_function_from_index(barycentric_coordinates, row_index)
                * shape_function_from_index(barycentric_coordinates, column_index);
        }

        // also called f_base
        pub fn ansatz_function(
            nodal_values: &Array1<f64>,
            barycentric_coordinates: &Array1<f64>,
        ) -> f64 {
            return nodal_values.dot(&all_shape_functions(barycentric_coordinates));
        }
    }

    mod shape_func_3d {
        use ndarray::Array1;

        fn lagrange_1_function(barycentric_coordinates: &Array1<f64>, index: usize) -> f64 {
            if index > 3 {
                panic!("Illegal Index given");
            };
            return barycentric_coordinates[index];
        }

        // Knoten 1 - 4 [Index 0 - 3]
        fn lagrange_2_order_nodal(barycentric_coordinates: &Array1<f64>, index: usize) -> f64 {
            if index > 3 {
                panic!("Illegal Index given");
            }
            return lagrange_1_function(barycentric_coordinates, index)
                * (2. * lagrange_1_function(barycentric_coordinates, index) - 1.);
        }

        // Knoten 5 - 10 [Index 4 - 9]
        fn lagrange_2_order_edge(barycentric_coordinates: &Array1<f64>, index: usize) -> f64 {
            if index < 4 {
                panic!("Illegal Index given");
            }
            if index < 7 {
                // 4,5,6
                let index = index - 4;
                return 4.
                    * lagrange_1_function(barycentric_coordinates, index)
                    * lagrange_1_function(barycentric_coordinates, 3 - index);
            } else {
                // 7,8,9
                let index = index - 7;
                return 4.
                    * lagrange_1_function(barycentric_coordinates, index)
                    * lagrange_1_function(barycentric_coordinates, 3);
            }
        }

        pub fn shape_function_from_index(
            barycentric_coordinates: &Array1<f64>,
            index: usize,
        ) -> f64 {
            if index > 9 {
                panic!("Illegal Index encountered")
            }
            if index < 4 {
                return lagrange_2_order_nodal(barycentric_coordinates, index);
            } else {
                return lagrange_2_order_edge(barycentric_coordinates, index);
            }
        }

        pub fn all_shape_functions(barycentric_coordinates: &Array1<f64>) -> Array1<f64> {
            let mut res = Array1::<f64>::zeros([10]);
            for i in 0..10 {
                res[i] = shape_function_from_index(barycentric_coordinates, i);
            }
            return res;
        }

        // [N \otimes N]_{ij}
        pub fn dyadic_product_component(
            barycentric_coordinates: &Array1<f64>,
            row_index: usize,
            column_index: usize,
        ) -> f64 {
            return shape_function_from_index(barycentric_coordinates, row_index)
                * shape_function_from_index(barycentric_coordinates, column_index);
        }

        // also called f_base
        pub fn ansatz_function(
            nodal_values: &Array1<f64>,
            barycentric_coordinates: &Array1<f64>,
        ) -> f64 {
            return nodal_values.dot(&all_shape_functions(barycentric_coordinates));
        }
    }

    pub mod problem_2d_definition {
        use ndarray::Array2;
        use ndarray::array;
        use ndarray::Array1;

        use crate::integration_2d::domain::ResultTypeWrapper;
        use crate::integration_2d::domain::Simplex2DFunction;

        use super::phase_field::{phase_field_func, varsigma_func_diff1, varsigma_func_diff2};
        use super::shape_func_2d::{ansatz_function, dyadic_product_component};

        // jetzt die Einzelintegranden
        pub fn phase_field_func_diff2(
            nodal_values: &Array1<f64>,
            kreg: f64,
            l: f64,
            row_index: usize,
            column_index: usize,
            barycentric_coordinates: &Array1<f64>,
        ) -> f64 {
            let f_base = ansatz_function(nodal_values, barycentric_coordinates);
            phase_field_func(f_base, kreg, l)
                * (varsigma_func_diff1(f_base, kreg).powi(2) - varsigma_func_diff2(f_base, kreg))
                * dyadic_product_component(barycentric_coordinates, row_index, column_index)
        }

        pub struct PhaseFieldFuncMatrix2D {
            nodal_values: Array1<f64>,
            kreg: f64,
            l: f64
        }

        impl PhaseFieldFuncMatrix2D {
            pub fn new(
                nodal_values: Array1<f64>,
                kreg: f64,
                l: f64,
            ) -> Self {
                return Self {
                    nodal_values,
                    kreg,
                    l
                };
            }
        }

        impl Simplex2DFunction for PhaseFieldFuncMatrix2D {
            type Return = ResultTypeWrapper<Array2<f64>>;

            fn additive_neutral_element(&self) -> Self::Return {
                ResultTypeWrapper::new(Array2::zeros([6,6]))
            }

            fn function(
                &self,
                xi1: f64,
                xi2: f64,
                xi3: f64,
                simplex: &crate::integration_2d::Simplex2D,
            ) -> Self::Return {
                let barycentric = array![xi1, xi2, xi3];

                let mut mat = Array2::zeros([6,6]);
                for i in 0..6 {
                    for j in 0..6 {
                        mat[[i,j]] =phase_field_func_diff2(
                            &self.nodal_values,
                            self.kreg,
                            self.l,
                            i,
                            j,
                            &barycentric,
                        );
                    }
                }

                return ResultTypeWrapper::new(mat);
            }
        }

        pub struct PhaseFieldFuncDiff22D {
            nodal_values: Array1<f64>,
            kreg: f64,
            l: f64,
            column_index: usize,
            row_index: usize,
        }

        impl PhaseFieldFuncDiff22D {
            pub fn new(
                nodal_values: Array1<f64>,
                kreg: f64,
                l: f64,
                column_index: usize,
                row_index: usize,
            ) -> Self {
                return Self {
                    nodal_values,
                    kreg,
                    l,
                    column_index,
                    row_index,
                };
            }
        }

        impl Simplex2DFunction for PhaseFieldFuncDiff22D {
            type Return = ResultTypeWrapper<f64>;
            fn function(
                &self,
                xi1: f64,
                xi2: f64,
                xi3: f64,
                simplex: &crate::integration_2d::Simplex2D,
            ) -> Self::Return {
                let barycentric = array![xi1, xi2, xi3];
                return ResultTypeWrapper::new(phase_field_func_diff2(
                    &self.nodal_values,
                    self.kreg,
                    self.l,
                    self.row_index,
                    self.column_index,
                    &barycentric,
                ));
            }
        }
    }

    pub mod problem_3d_definition {

        use ndarray::array;
        use ndarray::Array1;

        use crate::integration_3d::Simplex3DFunction;

        use super::phase_field::{phase_field_func, varsigma_func_diff1, varsigma_func_diff2};
        use super::shape_func_3d::{ansatz_function, dyadic_product_component};

        // jetzt die Einzelintegranden
        pub fn phase_field_func_diff2(
            nodal_values: &Array1<f64>,
            kreg: f64,
            l: f64,
            row_index: usize,
            column_index: usize,
            barycentric_coordinates: &Array1<f64>,
        ) -> f64 {
            let f_base = ansatz_function(nodal_values, barycentric_coordinates);
            phase_field_func(f_base, kreg, l)
                * (varsigma_func_diff1(f_base, kreg).powi(2) - varsigma_func_diff2(f_base, kreg))
                * dyadic_product_component(barycentric_coordinates, row_index, column_index)
        }

        pub struct PhaseFieldFuncDiff23D {
            nodal_values: Array1<f64>,
            kreg: f64,
            l: f64,
            column_index: usize,
            row_index: usize,
        }

        impl PhaseFieldFuncDiff23D {
            pub fn new(
                nodal_values: Array1<f64>,
                kreg: f64,
                l: f64,
                column_index: usize,
                row_index: usize,
            ) -> Self {
                return Self {
                    nodal_values,
                    kreg,
                    l,
                    column_index,
                    row_index,
                };
            }
        }

        impl Simplex3DFunction for PhaseFieldFuncDiff23D {
            fn function(
                &self,
                xi1: f64,
                xi2: f64,
                xi3: f64,
                xi4: f64,
                simplex: &crate::integration_3d::Simplex3D,
            ) -> f64 {
                let barycentric = array![xi1, xi2, xi3, xi4];
                return phase_field_func_diff2(
                    &self.nodal_values,
                    self.kreg,
                    self.l,
                    self.row_index,
                    self.column_index,
                    &barycentric,
                );
            }
        }
    }
}

pub struct PhaseField2DFunction {
    pub weights: [f64; 6],
}

impl Simplex2DFunction for PhaseField2DFunction {
    type Return = ResultTypeWrapper<f64>;
    fn function(&self, xi1: f64, xi2: f64, xi3: f64, _simplex: &crate::domain::Simplex2D) -> Self::Return {
        let f_base = approx_func(self.weights, xi1, xi2, xi3);
        return ResultTypeWrapper::new(phase_field_func(f_base, 0.000001, 1.0));
    }
}
