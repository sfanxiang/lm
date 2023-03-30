use tch::{Tensor, Kind, IndexOp, NewAxis};
use tch::{nn, nn::Module, Device};

#[derive(Debug)]
struct NN {
    model: nn::Sequential,
}

impl NN {
    pub fn new(vs: &nn::Path) -> Self {
        let linear = nn::linear(vs / "model", 32, 32, Default::default());
        linear.bs.as_ref().unwrap().print();
        Self {
            model: nn::seq().add(linear),
        }
    }
}

impl nn::Module for NN {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.model.forward(xs)
    }
}

struct Attention {
    
}

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    /*
    let model = NN::new(&vs.root());

    let input = Tensor::of_slice(&[0f32; 32]);
    // input.print();
    let output = model.forward(&input);
    output.print();
    */

    let max_positions = 20;
    let mut causal_mask = Tensor::ones(&[max_positions, max_positions], (Kind::Bool, Device::Cpu));
    causal_mask = causal_mask.tril(0);
    causal_mask = causal_mask.i((NewAxis, NewAxis));
    dbg!(causal_mask.size());
    causal_mask.print();
    /*
    let a = Tensor::of_slice(&[3, 1, 4, 1, 5]); [[5,4,4,4,4]] // [1,5]
    let b = Tensor::of_slice(&[3, 1, 4, 1, 5]); // [5]
    let c = a.unsqueeze(1) * b.unsqueeze(0);
    c.print();
    */
}
