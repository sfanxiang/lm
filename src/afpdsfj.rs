use rust_bert::gpt2::{Gpt2Config};
use rust_bert::resources::{LocalResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use tch::{nn, Device};

use super::gpt2;

pub fn afpdsfj() {
    
    let weights_path = weights_resource.get_local_path()?;

    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
    )?;
    let config = Gpt2Config::from_file(config_path);
    let gpt2_model = gtp2::GPT2LMHeadModel::new(&vs.root());
    vs.load(weights_path)?;
}
