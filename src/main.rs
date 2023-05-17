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

    fn at(&self, r: usize, c: usize) -> f32 {
        self.elements[r * self.cols + c]
    }

    fn at_mut(&mut self, r: usize, c: usize) -> &mut f32 {
        &mut self.elements.to_mut()[r * self.cols + c]
    }

    fn fill(&mut self, val: f32) {
        for e in self.elements.to_mut() {
            *e = val;
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

    fn zero(&mut self) {
        for w in &mut self.weights {
            for e in w.elements.to_mut() {
                *e = 0.;
            }
        }
        for b in &mut self.biases {
            for e in b.elements.to_mut() {
                *e = 0.;
            }
        }
        for a in &mut self.activations {
            for e in a.elements.to_mut() {
                *e = 0.;
            }
        }
    }

    fn input(&self) -> &'a Mat {
        &self.activations[0]
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

    fn get_output_for(&'a mut self, input: &Mat) -> &'a Mat {
        let self_ptr: *mut _ = self;
        assert_eq!(input.cols, self.input().cols);
        let this = unsafe { &mut *self_ptr };
        this.input_mut().copy_from(input);
        let this = unsafe { &mut *self_ptr };
        this.forward();
        self.output()
    }

    fn backprop(&'a mut self, g: &mut NN<LAYERS>, input: &Mat, output: &Mat) {
        assert_eq!(input.rows, output.rows);
        let n = input.rows;
        assert_eq!(output.cols, self.output().cols);

        g.zero();

        // i - current sample
        // l - current layer
        // j - current activation
        // k - previous activation
        let self_ptr: *mut _ = self;

        for i in 0..n {
            {
                let this = unsafe { &mut *self_ptr };
                this.input_mut().copy_from(&input.row(i));
            }
            self.forward();

            let g_ptr: *mut _ = g;
            for j in 0..LAYERS {
                let g_mut = unsafe { &mut *g_ptr };
                g_mut.activations[j].fill(0.);
            }

            for j in 0..self.output().cols {
                let d = self.output().at(0, j) - output.at(i, j);
                let g_mut = unsafe { &mut *g_ptr };
                *g_mut.output_mut().at_mut(0, j) = d;
            }

            for l in (1..=LAYERS).rev() {
                for j in 0..self.activations[l].cols {
                    let a = self.activations[l].at(0, j);
                    let da = g.activations[l].at(0, j);
                    *g.biases[l - 1].at_mut(0, j) += 2. * da * a * (1. - a);
                    for k in 0..self.activations[l - 1].cols {
                        // j - weight matrix col
                        // k - weight matrix row
                        // pa is the activation of the previous layer
                        let pa = self.activations[l - 1].at(0, k);
                        let w = self.weights[l - 1].at(k, j);
                        *g.weights[l - 1].at_mut(k, j) += 2. * da * a * (1. - a) * pa;
                        *g.activations[l - 1].at_mut(0, k) += 2. * da * a * (1. - a) * w;
                    }
                }
            }
        }

        for i in 0..LAYERS {
            for j in 0..g.weights[i].rows {
                for k in 0..g.weights[i].cols {
                    *g.weights[i].at_mut(j, k) /= n as f32;
                }
            }
            for j in 0..g.biases[i].rows {
                for k in 0..g.biases[i].cols {
                    *g.biases[i].at_mut(j, k) /= n as f32;
                }
            }
        }
    }

    fn learn(&mut self, g: &NN<LAYERS>, rate: f32) {
        for i in 0..LAYERS {
            for j in 0..self.weights[i].rows {
                for k in 0..self.weights[i].cols {
                    *self.weights[i].at_mut(j, k) -= rate * g.weights[i].at(j, k);
                }
            }

            for j in 0..self.biases[i].rows {
                for k in 0..self.biases[i].cols {
                    *self.biases[i].at_mut(j, k) -= rate * g.biases[i].at(j, k);
                }
            }
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
    let nn_ptr: *mut _ = &mut nn;
    train(&mut nn, &training_input, &training_output);
    let nn_mut = unsafe { &mut *nn_ptr };
    let new_cost = nn_mut.cost(&training_input, &training_output);
    dbg!(new_cost);

    // test the neural net
    for x in 0..=1 {
        for y in 0..=1 {
            let nn_mut = unsafe { &mut *nn_ptr };
            let input = Mat {
                rows: 1,
                cols: 2,
                elements: vec![x as f32, y as f32].into(),
            };
            let output = nn_mut.get_output_for(&input);
            let output = output.at(0, 0);
            println!("{x} ^ {y} = {} ({output})", (output > 0.5) as u8);
        }
    }
}

fn train<'a, const LAYERS: usize>(
    mut nn: &'a mut NN<'a, LAYERS>,
    training_input: &'a Mat,
    training_output: &'a Mat,
) where
    [(); LAYERS + 1]: Sized,
{
    let nn_ptr: *mut _ = &mut nn;

    let mut g = nn.clone();
    nn.randomize(0., 1.);

    let orig_cost = nn.cost(&training_input, &training_output);
    dbg!(orig_cost);

    let learn_rate = 1.;
    for _ in 0..5000 {
        let nn_mut = unsafe { &mut *nn_ptr };
        nn_mut.backprop(&mut g, &training_input, &training_output);
        let nn_mut = unsafe { &mut *nn_ptr };
        nn_mut.learn(&g, learn_rate);
    }
}
