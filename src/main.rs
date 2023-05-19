#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(maybe_uninit_uninit_array)]

use mnist::{Mnist, MnistBuilder};
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

    // TODO: NN takes parametric activation function
    fn sigmoid(&mut self) {
        for e in self.elements.to_mut() {
            *e = 1. / (1. + (-*e).exp());
        }
    }

    fn softmax(&mut self) {
        let mut max = self.elements[0];
        for e in &self.elements[1..] {
            if *e > max {
                max = *e;
            }
        }
        let mut sum = 0.;
        for e in self.elements.to_mut() {
            *e = (*e - max).exp();
            sum += *e;
        }
        for e in self.elements.to_mut() {
            *e /= sum;
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

    fn cost(&'a mut self, input: &Mat, output: &Mat, chunk_size: usize) -> f32 {
        assert_eq!(input.rows, output.rows);
        assert_eq!(output.cols, self.output().cols);
        // let n = input.rows;

        let n_lo = rand::random::<usize>() % (input.rows - chunk_size);
        let n_hi = n_lo + chunk_size;
        let n = n_hi - n_lo;

        let mut c = 0.;
        let self_ptr: *mut _ = self;
        for training_i in n_lo..n_hi {
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
            if i < LAYERS - 1 {
                next_activation.softmax();
            } else {
                // activation for MNIST is softmax for the output layer
                next_activation.sigmoid();
            }
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
        let chunk_size = 1000;
        let n_lo = rand::random::<usize>() % (input.rows - chunk_size);
        let n_hi = n_lo + chunk_size;
        let n = n_hi - n_lo;
        assert_eq!(output.cols, self.output().cols);

        g.zero();

        // i - current sample
        // l - current layer
        // j - current activation
        // k - previous activation
        let self_ptr: *mut _ = self;

        for i in n_lo..n_hi {
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

fn get_cost_from_gradient<const LAYERS: usize>(nn: &NN<LAYERS>) -> f32
where
    [(); LAYERS + 1]: Sized,
{
    nn.output().elements.iter().map(|x| x * x).sum::<f32>() / (nn.output().cols as f32)
}

#[allow(unused)]
fn xor_input_output_net<'a>() -> (Mat<'a>, Mat<'a>, NN<'a, 2>) {
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
    let nn = NN::<2>::new([2, 2, 1]);
    (training_input, training_output, nn)
}

fn one_hot_encode(n: usize, i: usize) -> impl Iterator<Item = f32> {
    (0..n).map(move |j| if j == i { 1. } else { 0. })
}

fn mnist_input_output_nn<'a>() -> (Mat<'a>, Mat<'a>, NN<'a, 2>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let training_input = Mat {
        rows: 50_000,
        cols: 28 * 28,
        elements: trn_img.iter().map(|&byte| byte as f32 / 255.).collect(),
    };
    let training_output = Mat {
        rows: 50_000,
        cols: 10,
        elements: trn_lbl
            .iter()
            .flat_map(|&byte| one_hot_encode(10, byte as usize))
            .collect(),
    };
    let nn = NN::<2>::new([28 * 28, 28 * 28, 10]);
    (training_input, training_output, nn)
}

fn main() {
    // let (training_input, training_output, mut nn) = xor_input_output_net();
    let (training_input, training_output, mut nn) = mnist_input_output_nn();
    let nn_ptr: *mut _ = &mut nn;
    train(&mut nn, &training_input, &training_output);
    let nn_mut = unsafe { &mut *nn_ptr };
    let new_cost = nn_mut.cost(&training_input, &training_output, 1000);
    dbg!(new_cost);

    // so let's spot check one mnist test example
    let Mnist {
        tst_img, tst_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
    let tst_is = vec![1, 69, 420, 1337];
    for tst_i in tst_is {
        let input = Mat {
            rows: 1,
            cols: 28 * 28,
            elements: tst_img[tst_i * 28 * 28..(tst_i + 1) * 28 * 28]
                .iter()
                .map(|&byte| byte as f32 / 255.)
                .collect(),
        };
        let nn_mut = unsafe { &mut *nn_ptr };
        let output = nn_mut.get_output_for(&input);
        let output = output
            .elements
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        // print out in ascii art, the image
        for i in 0..28 {
            for j in 0..28 {
                let pixel = input.at(0, i * 28 + j);
                print!(
                    "{}",
                    if pixel > 0.5 {
                        "██"
                    } else if pixel > 0.25 {
                        "▓▓"
                    } else if pixel > 0.125 {
                        "▒▒"
                    } else if pixel > 0.0625 {
                        "░░"
                    } else {
                        "  "
                    }
                );
            }
            println!();
        }
        println!("expected: {}, got: {}", tst_lbl[tst_i], output);
    }

    // // test the neural net
    // for x in 0..=1 {
    //     for y in 0..=1 {
    //         let nn_mut = unsafe { &mut *nn_ptr };
    //         let input = Mat {
    //             rows: 1,
    //             cols: 2,
    //             elements: vec![x as f32, y as f32].into(),
    //         };
    //         let output = nn_mut.get_output_for(&input);
    //         let output = output.at(0, 0);
    //         println!("{x} ^ {y} = {} ({output})", (output > 0.5) as u8);
    //     }
    // }
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

    let orig_cost = nn.cost(&training_input, &training_output, 1000);
    dbg!(orig_cost);

    let learn_rate = 1.;
    let num_iterations = 50;
    for i in 0..num_iterations {
        print!("iteration: {i}: ");

        let nn_mut = unsafe { &mut *nn_ptr };
        nn_mut.backprop(&mut g, &training_input, &training_output);

        let cost_after = get_cost_from_gradient(&g);
        println!("cost: {cost_after}");

        let nn_mut = unsafe { &mut *nn_ptr };
        nn_mut.learn(&g, learn_rate);
    }
}
