use core::borrow::Borrow;
use tch::{
    nn,
    nn::{ModuleT, Path},
    Device,
};
use tch::{IndexOp, Kind, NewAxis, Tensor};

#[derive(Debug)]
struct Dropout {
    p: f64,
}

impl Dropout {
    pub fn new<'a, T: Borrow<Path<'a>>>(_vs: T, p: f64) -> Self {
        // TODO
        Self { p }
    }
}

impl nn::ModuleT for Dropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.dropout(self.p, train)
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
    c_attn: nn::Linear,
    c_proj: nn::Linear,
    attn_dropout: Dropout,
    resid_dropout: Dropout,
}

impl Attention {
    fn new<'a, T: Borrow<Path<'a>>>(vs: T, layer_idx: i64) -> Self {
        let max_positions = 1024;

        let mut bias = Tensor::ones(&[max_positions, max_positions], (Kind::Bool, Device::Cpu));
        bias = bias.tril(0);
        bias = bias.i((NewAxis, NewAxis));
        let masked_bias = Tensor::full(&[], -1e4, (Kind::Float, Device::Cpu));

        let embed_dim = 768;
        let num_heads = 12;
        let head_dim = embed_dim / num_heads;
        let split_size = embed_dim;

        let scale_attn_weights = true;

        let scale_attn_by_inverse_layer_idx = false;
        let c_attn = nn::linear(
            vs.borrow() / "c_attn",
            embed_dim,
            embed_dim * 3,
            Default::default(),
        ); // TODO
        let c_proj = nn::linear(
            vs.borrow() / "c_proj",
            embed_dim,
            embed_dim,
            Default::default(),
        ); // TODO

        let attn_dropout = Dropout::new(vs.borrow() / "attn_dropout", 0.1);
        let resid_dropout = Dropout::new(vs.borrow() / "resid_dropout", 0.1);

        Self {
            bias,
            masked_bias,
            embed_dim,
            num_heads,
            head_dim,
            split_size,
            scale_attn_weights,
            scale_attn_by_inverse_layer_idx,
            layer_idx,
            c_attn,
            c_proj,
            attn_dropout,
            resid_dropout,
        }
    }

    fn attn(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        let mut attn_weights = query.matmul(&key.transpose(-1, -2));

        if self.scale_attn_weights {
            attn_weights = (&attn_weights)
                / tch::Tensor::full(
                    &[],
                    (*value.size().last().unwrap() as f64).sqrt(),
                    (attn_weights.kind(), attn_weights.device()),
                );
        }

        if self.scale_attn_by_inverse_layer_idx {
            attn_weights = (&attn_weights) / ((self.layer_idx + 1) as f64)
        }
        let query_length = query.size()[query.size().len() - 2];
        let key_length = query.size()[key.size().len() - 2];
        let causal_mask = self
            .bias
            .i((.., .., key_length - query_length..key_length, key_length));
        let mask_value = -1e10;
        let mask_value = tch::Tensor::full(
            &[],
            mask_value,
            (attn_weights.kind(), attn_weights.device()),
        );
        let mut attn_weights = attn_weights.where_self(&causal_mask, &mask_value);

        if let Some(attention_mask) = attention_mask {
            attn_weights = attn_weights + attention_mask;
        }
        let attn_output = attn_weights.matmul(&value);

        (attn_output, attn_weights)
    }

    fn split_heads(&self, tensor: &Tensor, num_heads: i64, attn_head_size: i64) -> Tensor {
        let mut new_shape = tensor.size();
        new_shape.pop();
        new_shape.push(num_heads);
        new_shape.push(attn_head_size);
        let tensor = tensor.reshape(&new_shape[..]);
        tensor.permute(&[0, 2, 1, 3]) // (batch, head, seq_length, head_features)
    }

    fn merge_heads(&self, tensor: &Tensor, num_heads: i64, attn_head_size: i64) -> Tensor {
        let tensor = tensor.permute(&[0, 2, 1, 3]).contiguous();
        let mut new_shape = tensor.size();
        new_shape.pop();
        new_shape.pop();
        new_shape.push(num_heads * attn_head_size);
        tensor.reshape(&new_shape[..])
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        let mut split = self.c_attn.forward_t(hidden_states, train);
        let mut split = split.split(self.split_size, 2);
        let mut value = split.pop().unwrap();
        let mut key = split.pop().unwrap();
        let mut query = split.pop().unwrap();

        query = self.split_heads(&query, self.num_heads, self.head_dim);
        key = self.split_heads(&key, self.num_heads, self.head_dim);
        value = self.split_heads(&value, self.num_heads, self.head_dim);

        let (mut attn_output, attn_weights) = self.attn(&query, &key, &value, attention_mask);

        attn_output = self.merge_heads(&attn_output, self.num_heads, self.head_dim);
        attn_output = self.c_proj.forward_t(&attn_output, train);
        attn_output = self.resid_dropout.forward_t(&attn_output, train);

        attn_output
    }
}

#[derive(Debug)]
struct NewGELUActivation {}

impl NewGELUActivation {
    pub fn new<'a, T: Borrow<Path<'a>>>(_vs: T) -> Self {
        Self {}
    }
}

impl nn::ModuleT for NewGELUActivation {
    fn forward_t(&self, input: &Tensor, _train: bool) -> Tensor {
        0.5 * input
            * (1.0
                + (2.0f64 / core::f64::consts::PI).sqrt()
                    * (input + 0.044715f64 * input.pow_tensor_scalar(3.0f64)).tanh())
    }
}

#[derive(Debug)]
struct MLP {
    c_fc: nn::Linear,
    c_proj: nn::Linear,
    act: NewGELUActivation,
    dropout: Dropout,
}

impl MLP {
    fn new<'a, T: Borrow<Path<'a>>>(vs: T, intermediate_size: i64, layer_idx: i64) -> Self {
        let embed_dim = 768;
        let c_fc = nn::linear(
            vs.borrow() / "c_fc",
            embed_dim,
            intermediate_size,
            Default::default(),
        ); // TODO
        let c_proj = nn::linear(
            vs.borrow() / "c_proj",
            intermediate_size,
            embed_dim,
            Default::default(),
        ); // TODO

        let act = NewGELUActivation::new(vs.borrow());
        let dropout = Dropout::new(vs.borrow() / "resid_dropout", 0.1);

        Self{
            c_fc, 
            c_proj,
            act,
            dropout
        }
    }
    
    fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        let mut hidden_states = self.c_fc.forward_t(hidden_states);
        hidden_states = self.act.forward_t(hidden_states)
        hidden_states = self.c_proj.forward_t(hidden_states)
        hidden_states = self.dropout.forward_t(hidden_states)
    }
}
/*
struct Block {
    ln_1: tch::nn::LayerNorm,
    attn: Attention,
    ln_2: tch::nn::LayerNorm,
    mlp:
}
impl Block {
    fn new() -> Self {
        let hidden_size = 768;
        let inner_dim = 4*hidden_size;//Because n_inner == None in config, inner_dim = 4*hidden_size
        let
    }
}
*/

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);

    let hidden_states = Tensor::zeros(&[1, 8, 768], (Kind::Float, Device::Cpu));

    let attn = Attention::new(&vs.root(), 0);
    let _output = attn.forward_t(&hidden_states, None, false);
}
