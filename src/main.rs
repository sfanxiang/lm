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
struct GPT2Attention {
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

impl GPT2Attention {
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
struct GPT2MLP {
    c_fc: nn::Linear,
    c_proj: nn::Linear,
    act: NewGELUActivation,
    dropout: Dropout,
}

impl GPT2MLP {
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, intermediate_size: i64) -> Self {
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

        Self {
            c_fc,
            c_proj,
            act,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, train: bool) -> Tensor {
        let mut hidden_states = self.c_fc.forward_t(hidden_states, train);
        hidden_states = self.act.forward_t(&hidden_states, train);
        hidden_states = self.c_proj.forward_t(&hidden_states, train);
        hidden_states = self.dropout.forward_t(&hidden_states, train);
        hidden_states
    }
}

struct GPT2Block {
    ln_1: nn::LayerNorm,
    attn: GPT2Attention,
    ln_2: nn::LayerNorm,
    mlp: GPT2MLP,
}

impl GPT2Block {
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, layer_idx: i64) -> Self {
        let hidden_size = 768;
        let inner_dim = 4 * hidden_size;
        let layer_norm_epsilon = 1e-5;

        let mut layer_norm_config: nn::LayerNormConfig = Default::default();
        layer_norm_config.eps = layer_norm_epsilon;
        let ln_1 = nn::layer_norm(vs.borrow(), vec![hidden_size], layer_norm_config);
        let attn = GPT2Attention::new(vs.borrow(), layer_idx);
        let ln_2 = nn::layer_norm(vs.borrow(), vec![hidden_size], layer_norm_config);

        let mlp = GPT2MLP::new(vs.borrow(), inner_dim);

        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        let residual = hidden_states;
        let mut hidden_states = self.ln_1.forward_t(hidden_states, train);
        let attn_output = self.attn.forward_t(&hidden_states, attention_mask, train);
        hidden_states = attn_output + residual;

        let residual = &hidden_states;
        let mut hidden_states = self.ln_2.forward_t(&hidden_states, train);
        let feed_forward_hidden_states = self.mlp.forward_t(&hidden_states, train);
        hidden_states = residual + feed_forward_hidden_states;

        return hidden_states;
    }
}

struct GPT2Model {
    embed_dim: i64,
    wte: nn::Embedding,
    wpe: nn::Embedding,
    drop: Dropout,
    h: Vec<GPT2Block>,
    ln_f: nn::LayerNorm,
}

impl GPT2Model {
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T) -> Self {
        let embed_dim = 768;

        let vocab_size = 50257;
        let max_position_embeddings = 1024;

        let wte = nn::embedding(vs.borrow(), vocab_size, embed_dim, Default::default());
        let wpe = nn::embedding(
            vs.borrow(),
            max_position_embeddings,
            embed_dim,
            Default::default(),
        );

        let embd_pdrop = 0.1;
        let drop = Dropout::new(vs.borrow(), embd_pdrop);

        let num_hidden_layers: i64 = 12;
        let mut h = Vec::with_capacity(num_hidden_layers as usize);
        for i in 0..num_hidden_layers {
            h.push(GPT2Block::new(vs.borrow(), i));
        }

        let layer_norm_epsilon = 1e-5;
        let mut layer_norm_config: nn::LayerNormConfig = Default::default();
        layer_norm_config.eps = layer_norm_epsilon;
        let ln_f = nn::layer_norm(vs.borrow(), vec![embed_dim], layer_norm_config);

        Self {
            embed_dim,
            wte,
            wpe,
            drop,
            h,
            ln_f,
        }
    }
}

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);

    let hidden_states = Tensor::zeros(&[1, 8, 768], (Kind::Float, Device::Cpu));

    let attn = GPT2Attention::new(&vs.root(), 0);
    let _output = attn.forward_t(&hidden_states, None, false);
}
