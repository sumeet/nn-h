#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(maybe_uninit_uninit_array)]

use std::mem::MaybeUninit;

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
}

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
    fn new(arch: [usize; LAYERS - 1]) -> Self {
        let mut activations: [MaybeUninit<Mat>; LAYERS + 1] =
            unsafe { MaybeUninit::uninit_array() };
        let activations: [Mat; LAYERS + 1] = unsafe {
            let ptr = &mut activations as *mut _ as *mut [Mat; LAYERS + 1];
            std::ptr::read(ptr)
        };

        let mut weights: [MaybeUninit<Mat>; LAYERS] = unsafe { MaybeUninit::uninit_array() };
        let weights: [Mat; LAYERS] = unsafe {
            let ptr = &mut weights as *mut _ as *mut [Mat; LAYERS];
            std::ptr::read(ptr)
        };

        let mut biases: [MaybeUninit<Mat>; LAYERS] = unsafe { MaybeUninit::uninit_array() };
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

    println!("Hello, world!");
}
