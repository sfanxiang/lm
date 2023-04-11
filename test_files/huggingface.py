import transformers
import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
input = tokenizer('<|endoftext|>I wonder how static masses emit gravitational waves to propogate their', return_tensors='pt', add_special_tokens=False)
model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
bad_id = tokenizer.encode('\n')[0]
times = []

for i in range(20):
    start_time = time.time()
    torch.manual_seed(0)
    result = model.generate(input['input_ids'], do_sample=True, topk = 50, bad_words_ids = [[bad_id]], temperature = 1, max_new_tokens = (i+1)*5)
    #print(tokenizer.batch_decode(result))
    times += [time.time() - start_time]
print(times)
