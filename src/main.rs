mod gpt2;

use self::gpt2::GPT2LMHeadModel;
use clap::Parser;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use std::io::Write;
use tch::{nn, Device, IndexOp, NewAxis, Tensor};

#[derive(Parser, Debug)]
struct Args {
    #[clap(long, default_value = "vocab.json")]
    vocab: String,
    #[clap(long, default_value = "merges.txt")]
    merges: String,
    #[clap(long, default_value = "model.ot")]
    model: String,
    #[clap(long)]
    sentence: String,
    #[clap(long, default_value = "0")]
    seed: i64,
    #[clap(long, default_value = "50")]
    topk: i64,
    #[clap(long, default_value = "1.0")]
    temperature: f64,
    #[clap(long, default_value = "20")]
    max_new_tokens: i64,
}

fn main() {
    let args = Args::parse();

    let device = Device::Cpu;

    let tokenizer = Gpt2Tokenizer::from_file(args.vocab, args.merges, false).unwrap();

    let weights_path = std::path::PathBuf::from(args.model);
    let mut vs = nn::VarStore::new(device);
    let model = GPT2LMHeadModel::new(&vs.root());
    vs.load(weights_path).unwrap();

    let new_sentence = args.sentence;
    print!("{}", &new_sentence[13..]);
    std::io::stdout().flush().unwrap();

    let bad_id = tokenizer
        .encode("\nX", None, 512, &TruncationStrategy::LongestFirst, 0)
        .token_ids[0];

    tch::manual_seed(args.seed);

    let input = tokenizer.encode(
        &new_sentence,
        None,
        512,
        &TruncationStrategy::LongestFirst,
        0,
    );

    let mut token_ids = input.token_ids;

    for _i in 0..args.max_new_tokens {
        let input_ids = Tensor::of_slice(&token_ids[..]).i(NewAxis);

        let output = model.forward_t(&input_ids, None, None, false);
        let mut logits = output.logits;
        //logits.i((.., .., ..4)).print();
        logits = logits.i((.., -1)) * args.temperature;

        //logits.i((.., bad_id)).copy_(&Tensor::of_slice(&[-1e10]));

        // topk
        // logits: (batch, num_tokens)
        let topk = if args.topk <= logits.size()[1] {
            args.topk
        } else {
            logits.size()[1]
        };
        let (topk_values, _) = logits.topk(topk, -1, true, true);
        // topk_values: (batch, topk)
        let topk_thres = topk_values.i((.., topk - 1, NewAxis));
        logits =
            logits.ge_tensor(&topk_thres) * (&logits) + logits.lt_tensor(&topk_thres) * (-1e10);

        let probs = logits.softmax(-1, logits.kind());
        let token = probs.multinomial(1, true);

        //let token = logits.argmax(Some(-1), false);

        let mut token_scalar = [0i64];
        token.copy_data(&mut token_scalar[..], 1);
        let token_scalar = token_scalar[0];
        let str_to_add = tokenizer.decode(&[token_scalar], false, false);
        print!("{}", str_to_add);
        std::io::stdout().flush().unwrap();

        token_ids.push(token_scalar);
        //new_sentence += &str_to_add[..];
    }
    print!("{}", "\n");
}
