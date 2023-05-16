#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(maybe_uninit_uninit_array)]

use std::mem::MaybeUninit;

#[derive(Debug, Clone)]
struct Mat {
    rows: usize,
    cols: usize,
    elements: Vec<f32>,
}

impl Mat {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            elements: vec![0.; rows * cols],
        }
    }

    // TODO: don't copy the data, but use references... how? idk in rust, maybe Cow
    fn row(&self, i: usize) -> Mat {
        todo!()
        // assert!(i < self.rows);
        // let mut row = Mat::new(1, self.cols);
        // for j in 0..self.cols {
        //     row.elements[j] = self.elements[i * self.cols + j];
        // }
        // row
    }
}

#[derive(Debug, Clone)]
struct NN<const LAYERS: usize>
where
    [(); LAYERS + 1]: Sized,
{
    weights: [Mat; LAYERS],
    biases: [Mat; LAYERS],
    activations: [Mat; LAYERS + 1],
}

impl<const LAYERS: usize> NN<LAYERS>
where
    [(); LAYERS + 1]: Sized,
{
    fn new(arch: [usize; LAYERS + 1]) -> Self {
        let mut activations: [MaybeUninit<Mat>; LAYERS + 1] =
            unsafe { MaybeUninit::uninit_array() };
        let mut weights: [MaybeUninit<Mat>; LAYERS] = unsafe { MaybeUninit::uninit_array() };
        let mut biases: [MaybeUninit<Mat>; LAYERS] = unsafe { MaybeUninit::uninit_array() };

        activations[0] = MaybeUninit::new(Mat::new(1, arch[0]));

        for i in 1..(LAYERS + 1) {
            weights[i - 1] = MaybeUninit::new(Mat::new(
                unsafe { activations[i - 1].assume_init_ref() }.cols,
                arch[i],
            ));
            biases[i - 1] = MaybeUninit::new(Mat::new(1, arch[i]));
            activations[i] = MaybeUninit::new(Mat::new(1, arch[i]));
        }

        let activations: [Mat; LAYERS + 1] = unsafe {
            let ptr = &mut activations as *mut _ as *mut [Mat; LAYERS + 1];
            std::ptr::read(ptr)
        };
        let weights: [Mat; LAYERS] = unsafe {
            let ptr = &mut weights as *mut _ as *mut [Mat; LAYERS];
            std::ptr::read(ptr)
        };
        let biases: [Mat; LAYERS] = unsafe {
            let ptr = &mut biases as *mut _ as *mut [Mat; LAYERS];
            std::ptr::read(ptr)
        };

        Self {
            weights,
            biases,
            activations,
        }
    }

    fn input_mut(&mut self) -> &mut Mat {
        &mut self.activations[0]
    }

    fn output(&self) -> &Mat {
        &self.activations[LAYERS]
    }

    fn randomize(&mut self, lo: f32, hi: f32) {
        for w in &mut self.weights {
            for e in &mut w.elements {
                *e = rand::random::<f32>() * (hi - lo) + lo;
            }
        }
        for b in &mut self.biases {
            for e in &mut b.elements {
                *e = rand::random::<f32>() * (hi - lo) + lo;
            }
        }
        for a in &mut self.activations {
            for e in &mut a.elements {
                *e = rand::random::<f32>() * (hi - lo) + lo;
            }
        }
    }

    // fn cost(&self, input: Mat, output: Mat) -> f32 {
    //     assert_eq!(input.rows, output.rows);
    //     assert_eq!(output.cols, self.output().cols);
    //     let n = input.rows;
    //
    //     let c = 0.;
    //     for i in 0..n {
    //         let x = input.row(i);
    //         let y = output.row(i);
    //         let input = self.input_mut();
    //         *input = x;
    //     }
    //     c / (n as f32)
    // }
}

fn main() {
    let training_input = Mat {
        rows: 4,
        cols: 2,
        elements: vec![
            0., 0., //
            0., 1., //
            1., 0., //
            1., 1., //
        ],
    };
    let training_output = Mat {
        rows: 4,
        cols: 1,
        elements: vec![
            0., //
            1., //
            1., //
            0., //
        ],
    };

    let mut nn = NN::<2>::new([2, 2, 1]);
    let mut g = nn.clone();
    nn.randomize(0., 1.);
    dbg!(nn);
}
