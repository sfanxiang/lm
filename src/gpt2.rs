use core::borrow::Borrow;
use tch::{
    nn,
    nn::{ModuleT, Path},
    Device,
};
use tch::{IndexOp, Kind, NewAxis, Tensor};

#[derive(Debug)]
pub struct Conv1D {
    weight: Tensor,
    bias: Tensor,
}

impl Conv1D {
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, nf: i64, nx: i64) -> Self {
        let weight = vs.borrow().var(
            "weight",
            &[nx, nf],
            nn::Init::Randn {
                mean: 0.,
                stdev: 0.02,
            },
        );
        let bias = vs.borrow().var("bias", &[nf], nn::Init::Const(0.));
        Self { weight, bias }
    }
}

impl nn::ModuleT for Conv1D {
    fn forward_t(&self, x: &Tensor, _train: bool) -> Tensor {
        let mut size_out = x.size();
        *size_out.last_mut().unwrap() = *self.weight.size().last().unwrap();
        let mut x = self
            .bias
            .addmm(&x.reshape(&[-1, *x.size().last().unwrap()]), &self.weight);
        x = x.reshape(&size_out[..]);
        x
    }
}

#[derive(Debug)]
pub struct Dropout {
    p: f64,
}

impl Dropout {
    pub fn new<'a, T: Borrow<Path<'a>>>(_vs: T, p: f64) -> Self {
        Self { p }
    }
}

impl nn::ModuleT for Dropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.dropout(self.p, train)
    }
}

#[derive(Debug)]
pub struct NewGELUActivation {}

impl NewGELUActivation {
    pub fn new<'a, T: Borrow<Path<'a>>>(_vs: T) -> Self {
        Self {}
    }
}

impl nn::ModuleT for NewGELUActivation {
    fn forward_t(&self, input: &Tensor, _train: bool) -> Tensor {
        0.5 * input
            * (1.0
                + ((2.0f64 / core::f64::consts::PI).sqrt()
                    * (input + 0.044715f64 * input.pow_tensor_scalar(3.0f64)))
                .tanh())
    }
}

#[derive(Debug)]
pub struct BaseModelOutput {
    pub hidden_states: Tensor,
    pub past_key_values: Option<Vec<(Tensor, Tensor)>>,
}

#[derive(Debug)]
pub struct CausalLMOutput {
    pub logits: Tensor,
}

#[derive(Debug)]
pub struct GPT2Attention {
    bias: Tensor,
    #[allow(dead_code)]
    masked_bias: Tensor,
    #[allow(dead_code)]
    embed_dim: i64,
    num_heads: i64,
    head_dim: i64,
    split_size: i64,
    scale_attn_weights: bool,
    scale_attn_by_inverse_layer_idx: bool,
    layer_idx: i64,
    c_attn: Conv1D,
    c_proj: Conv1D,
    attn_dropout: Dropout,
    resid_dropout: Dropout,
}

impl GPT2Attention {
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, layer_idx: i64) -> Self {
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
        let c_attn = Conv1D::new(vs.borrow() / "c_attn", embed_dim * 3, embed_dim);
        let c_proj = Conv1D::new(vs.borrow() / "c_proj", embed_dim, embed_dim);

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

    pub fn attn(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
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
        let causal_mask =
            self.bias
                .i((.., .., key_length - query_length..key_length, ..key_length));
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

        attn_weights = attn_weights.softmax(-1, attn_weights.kind());
        attn_weights = self.attn_dropout.forward_t(&attn_weights, train);

        let attn_output = attn_weights.matmul(&value);

        (attn_output, attn_weights)
    }

    pub fn split_heads(&self, tensor: &Tensor, num_heads: i64, attn_head_size: i64) -> Tensor {
        let mut new_shape = tensor.size();
        new_shape.pop();
        new_shape.push(num_heads);
        new_shape.push(attn_head_size);
        let tensor = tensor.reshape(&new_shape[..]);
        tensor.permute(&[0, 2, 1, 3]) // (batch, head, seq_length, head_features)
    }

    pub fn merge_heads(&self, tensor: &Tensor, num_heads: i64, attn_head_size: i64) -> Tensor {
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
        layer_past: Option<(&Tensor, &Tensor)>,
        attention_mask: Option<&Tensor>,
        use_cache: bool,
        train: bool,
    ) -> (Tensor, Option<(Tensor, Tensor)>) {
        let split = self.c_attn.forward_t(hidden_states, train);
        let mut split = split.split(self.split_size, 2);
        let mut value = split.pop().unwrap();
        let mut key = split.pop().unwrap();
        let mut query = split.pop().unwrap();

        query = self.split_heads(&query, self.num_heads, self.head_dim);
        key = self.split_heads(&key, self.num_heads, self.head_dim);
        value = self.split_heads(&value, self.num_heads, self.head_dim);

        match layer_past {
            Some((key_from_lp, value_from_lp)) => {
                let past_key = key_from_lp;
                let past_value = value_from_lp;
                key = Tensor::cat(&[past_key, &key], -2);
                value = Tensor::cat(&[past_value, &value], -2);
            }
            None(_) => (),
        }
        let present =
        if use_cache {
            Some((key,value))
        }
        else {
            None
        }; 

        let (mut attn_output, _attn_weights) =
            self.attn(&query, &key, &value, attention_mask, train);

        attn_output = self.merge_heads(&attn_output, self.num_heads, self.head_dim);
        attn_output = self.c_proj.forward_t(&attn_output, train);
        attn_output = self.resid_dropout.forward_t(&attn_output, train);

        (attn_output, present)
    }
}

#[derive(Debug)]
pub struct GPT2MLP {
    c_fc: Conv1D,
    c_proj: Conv1D,
    act: NewGELUActivation,
    dropout: Dropout,
}

impl GPT2MLP {
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, intermediate_size: i64) -> Self {
        let embed_dim = 768;
        let c_fc = Conv1D::new(vs.borrow() / "c_fc", intermediate_size, embed_dim);
        let c_proj = Conv1D::new(vs.borrow() / "c_proj", embed_dim, intermediate_size);

        let act = NewGELUActivation::new(vs.borrow() / "act");
        let dropout = Dropout::new(vs.borrow() / "dropout", 0.1);

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

pub struct GPT2Block {
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
        let ln_1 = nn::layer_norm(vs.borrow() / "ln_1", vec![hidden_size], layer_norm_config);
        let attn = GPT2Attention::new(vs.borrow() / "attn", layer_idx);
        let ln_2 = nn::layer_norm(vs.borrow() / "ln_2", vec![hidden_size], layer_norm_config);

        let mlp = GPT2MLP::new(vs.borrow() / "mlp", inner_dim);

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
        layer_past: Option<(&Tensor, &Tensor)>,
        attention_mask: Option<&Tensor>,
        use_cache: bool,
        train: bool,
    ) -> (Tensor, Option<(Tensor, Tensor)>) {
        let residual = hidden_states;
        let mut hidden_states = self.ln_1.forward_t(hidden_states, train);
        let attn_outputs = self
            .attn
            .forward_t(&hidden_states, layer_past, attention_mask, use_cache, train);
        let attn_output = attn_outputs.0;
        let outputs = (attn_outputs.1,);
        hidden_states = attn_output + residual;

        let residual = &hidden_states;
        let mut hidden_states = self.ln_2.forward_t(&hidden_states, train);
        let feed_forward_hidden_states = self.mlp.forward_t(&hidden_states, train);
        hidden_states = residual + feed_forward_hidden_states;

        (hidden_states, outputs.0)
    }
}

pub struct GPT2Model {
    #[allow(dead_code)]
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

        let wte = nn::embedding(
            vs.borrow() / "wte",
            vocab_size,
            embed_dim,
            Default::default(),
        );
        let wpe = nn::embedding(
            vs.borrow() / "wpe",
            max_position_embeddings,
            embed_dim,
            Default::default(),
        );

        let embd_pdrop = 0.1;
        let drop = Dropout::new(vs.borrow() / "drop", embd_pdrop);

        let num_hidden_layers: i64 = 12;
        let mut h = Vec::with_capacity(num_hidden_layers as usize);
        for i in 0..num_hidden_layers {
            h.push(GPT2Block::new(vs.borrow() / "h" / i.to_string(), i));
        }

        let layer_norm_epsilon = 1e-5;
        let mut layer_norm_config: nn::LayerNormConfig = Default::default();
        layer_norm_config.eps = layer_norm_epsilon;
        let ln_f = nn::layer_norm(vs.borrow() / "ln_f", vec![embed_dim], layer_norm_config);

        Self {
            embed_dim,
            wte,
            wpe,
            drop,
            h,
            ln_f,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        mut past_key_values: Option<(&[(&Tensor, &Tensor)])>,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        use_cache: bool,
        train: bool,
    ) -> BaseModelOutput {
        
        let input_shape = input_ids.size();
        let input_ids = input_ids.reshape(&[-1, *input_shape.last().unwrap()]);
        let batch_size = input_ids.size()[0];

        let device = input_ids.device();

        let position_ids = position_ids.map(|x| x.reshape(&[-1, *input_shape.last().unwrap()]));
        match past_key_values {
            None(_) => {
                let past_length = 0;
            }
            Some(x) => {
                let past_length = x[0][0].size()[-2];
            }
        }
        let position_ids = position_ids.unwrap_or_else(|| {
            let position_ids = Tensor::arange_start(
                past_length,
                *input_shape.last().unwrap() + past_length,
                (Kind::Int64, device),
            );
            position_ids
                .unsqueeze(0)
                .reshape(&[-1, *input_shape.last().unwrap()])
        });

        let attention_mask = attention_mask.map(|attention_mask| {
            let mut attention_mask = attention_mask.reshape(&[batch_size, -1]);
            attention_mask = attention_mask.i((.., NewAxis, NewAxis, ..));
            attention_mask = attention_mask.to_dtype(Kind::Float, false, false);
            attention_mask = (1.0 - attention_mask) * (-1e10);
            attention_mask
        });

        let inputs_embeds = self.wte.forward_t(&input_ids, train);
        let position_embeds = self.wpe.forward_t(&position_ids, train);
        let mut hidden_states = inputs_embeds + position_embeds;

        hidden_states = self.drop.forward_t(&hidden_states, train);

        let mut output_shape = input_shape;
        output_shape.push(*hidden_states.size().last().unwrap());

        let mut presents = 
        if use_cache {
            Some(Vec::with_capacity(self.h.len()))
        }
        else {
            None
        };


        for (_i, (block, layer_past)) in self.h.iter().zip(past_key_values.iter()).enumerate() {
            let outputs =
                block.forward_t(&hidden_states, layer_past, attention_mask.as_ref(), use_cache, train);
            hidden_states = outputs.0;
            if use_cache {
                presents.as_mut().unwrap().push(outputs.1.unwrap());
            }
        }

        hidden_states = self.ln_f.forward_t(&hidden_states, train);
        hidden_states = hidden_states.reshape(&output_shape[..]);

        BaseModelOutput { hidden_states, past_key_values: Some(presents)}
    }
}

pub struct GPT2LMHeadModel {
    transformer: GPT2Model,
}

impl GPT2LMHeadModel {
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T) -> Self {
        let transformer = GPT2Model::new(vs.borrow() / "transformer");
        Self { transformer }
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        train: bool,
    ) -> CausalLMOutput {
        let transformer_outputs =
            self.transformer
                .forward_t(input_ids, attention_mask, position_ids, train);
        let hidden_states = transformer_outputs.hidden_states;

        let weight = &self.transformer.wte.ws;
        let lm_logits = hidden_states.matmul(&weight.transpose(-2, -1));

        CausalLMOutput { logits: lm_logits }
    }
}
