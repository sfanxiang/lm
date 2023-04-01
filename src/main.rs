use tch::Tensor;
use tch::{nn, Device};

mod gpt2;
mod afpdsfj;
use gpt2::GPT2LMHeadModel;

fn main() {
    /*let vs = nn::VarStore::new(Device::Cpu);

    let input_ids = Tensor::of_slice(&[1, 2, 3]);

    let model = GPT2LMHeadModel::new(&vs.root());
    let outputs = model.forward_t(&input_ids, None, None, false);*/

    //outputs.logits.print();
    afpdsfj::afpdsfj();
}
