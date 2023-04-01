use rust_bert::gpt2::{Gpt2Config};
use rust_bert::resources::{LocalResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use tch::{nn, Device};

use super::gpt2;

pub fn afpdsfj() {
    let weights_path = std::path::PathBuf::from("rust_model.ot");

    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let gpt2_model = gpt2::GPT2LMHeadModel::new(&vs.root());
    vs.load(weights_path);
}
