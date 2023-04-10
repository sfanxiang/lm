use core::borrow::Borrow;
use lm::gpt2::GPT2LMHeadModel;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use std::io::Write;
use std::process::Command;
use std::time::{Duration, Instant};
use tch::{nn, Device, IndexOp, Kind, NewAxis, Tensor};

fn main() {
    let mut times_tokens: Vec<(f32, u32)> = Vec::new();
    let device = Device::Cpu;

    let tokenizer = Gpt2Tokenizer::from_file("vocab.json", "merges.txt", false).unwrap();

    let weights_path = std::path::PathBuf::from("model.ot");
    let mut vs = nn::VarStore::new(device);
    let model = GPT2LMHeadModel::new(&vs.root());
    vs.load(weights_path).unwrap();

    let new_sentence =
        "<|endoftext|>I wonder how static masses emit gravitational waves to propogate their";

    let bad_id = tokenizer
        .encode("\nX", None, 512, &TruncationStrategy::LongestFirst, 0)
        .token_ids[0];

    let input = tokenizer.encode(
        &new_sentence,
        None,
        512,
        &TruncationStrategy::LongestFirst,
        0,
    );

    for i in 0..10 {
        let start = Instant::now();
        tch::manual_seed(0);
        let mut past_key_values = None;
        let mut token_ids = input.token_ids.clone();
        //println!("Test {}", i);
        for j in 0..(i + 1) * 3 {
            let input_ids = if past_key_values.is_some() {
                Tensor::of_slice(&token_ids[token_ids.len() - 1..]).i(NewAxis)
            } else {
                Tensor::of_slice(&token_ids[..]).i(NewAxis)
            };

            let past_key_values_ref = past_key_values
                .as_ref()
                .map(|x: &Vec<(Tensor, Tensor)>| x.iter().map(|(k, v)| (k, v)).collect::<Vec<_>>());
            let output = model.forward_t(
                &input_ids,
                past_key_values_ref.as_ref().map(|x| &x[..]),
                None,
                None,
                true,
                false,
            );
            let mut logits = output.logits;
            past_key_values = output.past_key_values;
            //logits.i((.., .., ..4)).print();
            logits = logits.i((.., -1));

            //logits.i((.., bad_id)).copy_(&Tensor::of_slice(&[-1e10]));

            // topk
            // logits: (batch, num_tokens)
            let topk = if 50 <= logits.size()[1] {
                50
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

            let mut token_scalar = [0i64];
            token.copy_data(&mut token_scalar[..], 1);
            let token_scalar = token_scalar[0];
            let str_to_add = tokenizer.decode(&[token_scalar], false, false);
            //print!("{}", str_to_add);

            token_ids.push(token_scalar);
        }
        let duration = start.elapsed();
        times_tokens.push((duration.as_millis() as f32 / 1000., (i + 1) * 3));
    }

    println!("{:?}", times_tokens);
}
