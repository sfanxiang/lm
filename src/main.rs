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

#[derive(Debug)]
struct Attention {
    bias: Tensor,
    masked_bias: Tensor, 
    embed_dim: i64,
    num_heads: i64, 
    head_dim: i64, 
    split_size: i64, 
    scale_attn_weights: bool,
    scale_attn_by_inverse_layer_idx: bool, 
    layer_idx: i64,
    c_attn: nn::Conv1D,
    q_attn: nn::Conv1D,
    c_proj: nn::Conv1D,
}

impl Attention {
    fn attn(&self, query: &Tensor, key: &Tensor, value: &Tensor, attention_mask: Option<&Tensor>) 
    -> (Tensor, Tensor){
        let mut attn_weights = query.matmul(&key.transpose(-1, -2));
        
        if self.scale_attn_weights {
            attn_weights = (&attn_weights) / tch::Tensor::full(&[], (*value.size().last().unwrap() as f64).sqrt(), (attn_weights.kind(), attn_weights.device()));
        }

        if self.scale_attn_by_inverse_layer_idx {
            attn_weights = (&attn_weights) / ((self.layer_idx + 1) as f64) 
        }
        let sec_to_last_query_val = query.size()[query.size().len() - 2];
        let sec_to_last_key_val = query.size()[key.size().len() - 2];
        let (query_length, key_length) = (sec_to_last_query_val, sec_to_last_key_val);
        let causal_mask = self.bias.i((.., .., (key.size().len() - query.size().len()) as i64  .. key.size().len() as i64 , .. key.size().len() as i64 ));
        let mask_value = -1e10;
        let mask_value = tch::Tensor::full(&[], mask_value, (attn_weights.kind(), attn_weights.device()));
        let mut attn_weights = causal_mask.where_self(&attn_weights, &mask_value);

        if let Some(attention_mask) = attention_mask {
            attn_weights = attn_weights + attention_mask;
        }
        let attn_output = attn_weights.matmul(&value);

        return (attn_output, attn_weights)
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Tensor {
        todo!()
    }
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
