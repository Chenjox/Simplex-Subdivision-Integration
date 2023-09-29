use ndarray::Array2;

pub struct IntegratorDummy;

impl IntegratorDummy {
    pub fn get() -> Self {
        Self {}
    }
}

pub fn det3x3(mat3x3: &Array2<f64>) -> f64 {
    let mut sum = 0.0;
    for j in 0..3 {
        let mut prod = 1.0;
        for i in 0..3 {
            prod = prod * mat3x3[[i, (i + j) % 3]]
        }
        sum = sum + prod;
        let mut prod = -1.0;
        for i in 0..3 {
            prod = prod * mat3x3[[(2 - i), (i + j) % 3]]
        }
        sum = sum + prod;
    }
    return sum;
}

pub fn det4x4(mat4x4: &Array2<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        let val = mat4x4[[i, 0]];
        let mut mat3x3 = Array2::<f64>::zeros([3, 3]);
        for j in 0..3 {
            if j >= i {
                for k in 0..3 {
                    mat3x3[[j, k]] = mat4x4[[j + 1, k + 1]]
                }
            } else {
                for k in 0..3 {
                    mat3x3[[j, k]] = mat4x4[[j, k + 1]]
                }
            }
        }
        sum += (-1.0_f64).powi(i as i32) * val * det3x3(&mat3x3);
    }
    return sum;
}
