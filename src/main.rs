mod gpt2;

use self::gpt2::GPT2LMHeadModel;
use clap::Parser;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use std::io::Write;
use tch::{nn, Device, IndexOp, NewAxis, Tensor};

#[derive(Parser, Debug)]
struct Args {
    #[clap(long)]
    vocab: String,
    #[clap(long)]
    merges: String,
    #[clap(long)]
    model: String,
    #[clap(long)]
    sentence: String,
    #[clap(long)]
    max_new_tokens: i64,
}

fn main() {
    let args = Args::parse();

    let device = Device::Cpu;

    let tokenizer = Gpt2Tokenizer::from_file(args.vocab, args.merges, false).unwrap();

    let weights_path = std::path::PathBuf::from("rust_model.ot");
    let mut vs = nn::VarStore::new(device);
    let model = GPT2LMHeadModel::new(&vs.root());
    vs.load(weights_path).unwrap();

    let mut new_sentence = args.sentence;
    print!("{}", &new_sentence[13..]);
    std::io::stdout().flush().unwrap();
    for _i in 0..args.max_new_tokens {
        let input = tokenizer.encode(
            &new_sentence,
            None,
            512,
            &TruncationStrategy::LongestFirst,
            0,
        );
        let input_ids = Tensor::of_slice(&input.token_ids[..]).i(NewAxis);

        let output = model.forward_t(&input_ids, None, None, false);
        let logits = output.logits;
        let logits = logits.i((.., -1));
        let token = logits.argmax(Some(-1), false);

        let mut token_scalar = [0i64];
        token.copy_data(&mut token_scalar[..], 1);
        let token_scalar = token_scalar[0];
        let str_to_add = tokenizer.decode(&[token_scalar], false, false);
        print!("{}", str_to_add);
        std::io::stdout().flush().unwrap();
        new_sentence += &str_to_add[..];
    }
    print!("{}", "\n");
}
