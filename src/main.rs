#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(maybe_uninit_uninit_array)]

use std::borrow::Cow;
use std::mem::MaybeUninit;

#[derive(Debug, Clone)]
struct Mat<'a> {
    rows: usize,
    cols: usize,
    elements: Cow<'a, [f32]>,
}

impl<'a> Mat<'a> {
    fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            elements: vec![0.; rows * cols].into(),
        }
    }

    fn copy_from(&mut self, other: &Mat) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        self.elements.to_mut().copy_from_slice(&other.elements);
    }

    fn row(&'a self, i: usize) -> Mat<'a> {
        let els = &self.elements[(i * self.cols)..((i + 1) * self.cols)];
        Mat {
            rows: 1,
            cols: self.cols,
            elements: Cow::Borrowed(els),
        }
    }

    fn dot_from(&mut self, a: &Mat, b: &Mat) {
        assert_eq!(a.cols, b.rows);
        assert_eq!(self.rows, a.rows);
        assert_eq!(self.cols, b.cols);
        let elements = self.elements.to_mut();
        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut sum = 0.;
                for k in 0..a.cols {
                    sum += a.elements[i * a.cols + k] * b.elements[k * b.cols + j];
                }
                elements[i * self.cols + j] = sum;
            }
        }
    }

    fn add(&mut self, m: &Mat) {
        assert_eq!(self.rows, m.rows);
        assert_eq!(self.cols, m.cols);
        let elements = self.elements.to_mut();
        for i in 0..self.rows {
            for j in 0..self.cols {
                elements[i * self.cols + j] += m.elements[i * self.cols + j];
            }
        }
    }

    fn sigmoid(&mut self) {
        for e in self.elements.to_mut() {
            *e = 1. / (1. + (-*e).exp());
        }
    }
}

#[derive(Debug, Clone)]
struct NN<'a, const LAYERS: usize>
where
    [(); LAYERS + 1]: Sized,
{
    weights: [Mat<'a>; LAYERS],
    biases: [Mat<'a>; LAYERS],
    activations: [Mat<'a>; LAYERS + 1],
}

impl<'a, const LAYERS: usize> NN<'a, LAYERS>
where
    [(); LAYERS + 1]: Sized,
{
    fn new(arch: [usize; LAYERS + 1]) -> Self {
        let mut activations: [MaybeUninit<Mat>; LAYERS + 1] = MaybeUninit::uninit_array();
        let mut weights: [MaybeUninit<Mat>; LAYERS] = MaybeUninit::uninit_array();
        let mut biases: [MaybeUninit<Mat>; LAYERS] = MaybeUninit::uninit_array();

        activations[0] = MaybeUninit::new(Mat::zero(1, arch[0]));

        for i in 1..(LAYERS + 1) {
            weights[i - 1] = MaybeUninit::new(Mat::zero(
                unsafe { activations[i - 1].assume_init_ref() }.cols,
                arch[i],
            ));
            biases[i - 1] = MaybeUninit::new(Mat::zero(1, arch[i]));
            activations[i] = MaybeUninit::new(Mat::zero(1, arch[i]));
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

    fn input_mut(&mut self) -> &'a mut Mat {
        &mut self.activations[0]
    }

    fn output(&self) -> &Mat {
        &self.activations[LAYERS]
    }

    fn output_mut(&mut self) -> &'a mut Mat {
        &mut self.activations[LAYERS]
    }

    fn randomize(&mut self, lo: f32, hi: f32) {
        for w in &mut self.weights {
            for e in w.elements.to_mut() {
                *e = rand::random::<f32>() * (hi - lo) + lo;
            }
        }
        for b in &mut self.biases {
            for e in b.elements.to_mut() {
                *e = rand::random::<f32>() * (hi - lo) + lo;
            }
        }
        for a in &mut self.activations {
            for e in a.elements.to_mut() {
                *e = rand::random::<f32>() * (hi - lo) + lo;
            }
        }
    }

    fn cost(&'a mut self, input: &Mat, output: &Mat) -> f32 {
        assert_eq!(input.rows, output.rows);
        assert_eq!(output.cols, self.output().cols);
        let n = input.rows;

        let mut c = 0.;
        let self_ptr: *mut _ = self;
        for training_i in 0..n {
            let x = input.row(training_i);
            let y = output.row(training_i);
            {
                let this = unsafe { &mut *self_ptr };
                this.input_mut().copy_from(&x);
            }
            self.forward();
            for i in 0..self.output().cols {
                let d = self.output().elements[i] - y.elements[i];
                c += d * d;
            }
        }
        c / (n as f32)
    }

    fn forward(&mut self) {
        for i in 0..LAYERS {
            let (prev_layer, next_layer) = self.activations.split_at_mut(i + 1);
            let next_activation = &mut next_layer[0];
            let activation = &prev_layer[i];
            let weight = &self.weights[i];
            let bias = &self.biases[i];
            next_activation.dot_from(activation, weight);
            next_activation.add(bias);
            next_activation.sigmoid();
        }
    }
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
        ]
        .into(),
    };
    let training_output = Mat {
        rows: 4,
        cols: 1,
        elements: vec![
            0., //
            1., //
            1., //
            0., //
        ]
        .into(),
    };

    let mut nn = NN::<2>::new([2, 2, 1]);
    let mut g = nn.clone();
    nn.randomize(0., 1.);
    dbg!(&nn);
    dbg!(nn.cost(&training_input, &training_output));
}
