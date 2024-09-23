import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import sys
import nnsight
import math

# Define the Transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))
        self.transformer_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.transformer_layer(x, x, tgt_mask=mask, tgt_is_causal=True, memory_mask=mask, memory_is_causal=True)
        return self.fc(x)

    def generate(self, input_ids, max_length, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - len(input_ids[0])):
                #print(input_ids)
                outputs = self(input_ids)
                #outputs = self(torch.cat([input_ids, torch.zeros((1,1), dtype=torch.int).to(input_ids.device)], dim=-1))
                next_token_logits = outputs[:, -1, :] / temperature

                # Apply top-k and top-p filtering
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=50)

                # Apply softmax to convert logits to probabilities
                probs = F.softmax(filtered_logits, dim=-1)

                # Ensure no zeros in probabilities to avoid multinomial error
                probs2 = probs + 1e-8
                probs2 = probs2 / probs2.sum()  # Renormalize

                try:
                    next_token = torch.multinomial(probs2, num_samples=1)
                    #print('token prob', probs[-1,next_token[0,0]].item())
                except RuntimeError as e:
                    print(f"RuntimeError in multinomial sampling: {e}")
                    print(f"Probabilities min: {probs.min().item()}, max: {probs.max().item()}")
                    print(f"Probabilities sum: {probs.sum().item()}")
                    # Fallback: choose the most likely token
                    next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

# Helper function for nucleus sampling
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-10000000.0):
    try:
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits += indices_to_remove * filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        # Ensure at least one token is kept
        if (logits == filter_value).all():
            logits[..., 0] = 0

        return logits

    except Exception as e:
        print(f"Error in top_k_top_p_filtering: {e}")
        print(f"logits shape: {logits.shape}")
        print(f"logits min: {logits.min().item()}, max: {logits.max().item()}")
        print(f"top_k: {top_k}, top_p: {top_p}")
        # In case of error, return the original logits
        return logits


# This class basically follows the gated SAEs paper exactly, using the same names, mostly
class GSAEModel(nn.Module):
    def __init__(self, d_model, n_features, sparsity_penalty):
        super().__init__()
        self.n_features = n_features
        self.w_gate = nn.Parameter(torch.randn(n_features, d_model)) # (F,D)
        self.r_mag = nn.Parameter(torch.randn(n_features)) # (F)
        self.b_dec = nn.Parameter(torch.randn(d_model)) # (D)
        self.b_gate = nn.Parameter(torch.randn(n_features)) # (F)
        self.b_mag = nn.Parameter(torch.randn(n_features))  # (F)
        self.w_dec = nn.Parameter(torch.randn(d_model, n_features)) # (D,F)
        self.sparsity_penalty = sparsity_penalty

    # returns (encoded->decoded x, pi_gate) in order for the loss to work properly
    def forward(self, x):
        x2 = x - self.b_dec
        # this is W_gate(x2)
        # but because x2 is batched with batch first, we have to transpose and do the multiplication backwards
        wg_x = torch.matmul(x2, self.w_gate.transpose(0,1))
        # we also need W_mag(x2), where W_mag = exp(r_mag) * W_gate (* is componentwise distributing over rows)
        # now, because we're distributing r_mag over rows, and then we'll be matmul-ing by x2 which will dot x2 with each row and return a column vector,
        # we can actually reassociate and do exp(r_mag) * (W_gate(x2)), where * is componentwise
        wm_x = torch.exp(self.r_mag) * wg_x
        pi_gate = wg_x + self.b_gate
        pi_mag = wm_x + self.b_mag
        f_gate = torch.heaviside(pi_gate, torch.tensor(0.0)).detach() # trying to backward through heaviside gives a not implemented error
        f_mag = torch.relu(pi_mag)                                    # instead, we just want to ignore the gradient
        fx = f_gate * f_mag
        # also modified for batch-first
        xd = torch.matmul(fx, self.w_dec.transpose(0,1)) + self.b_dec
        return xd, pi_gate

    # returns loss, reconstruction loss because having the reconstruction loss separate is convenient
    def loss(self, x):
        xd, pi_gate = self(x)
        # reconstruction loss
        l_r = torch.square(x - xd).sum() # squared 2-norm
        # sparsity loss
        l_s = self.sparsity_penalty * torch.relu(pi_gate).sum()
        # auxiliary loss
        # also modified for batch-first
        l_a = torch.square(x - torch.matmul(torch.relu(pi_gate), self.w_dec.detach().transpose(0,1)) + self.b_dec.detach()).sum()
        return l_r + l_s + l_a, l_r


# JumpRelu stuff:
bandwidth = 0.001

def rectangle(x):
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)

class JumpReluActivation(torch.autograd.Function):
    @staticmethod
    def forward(cxt, x, theta):
        cxt.save_for_backward(theta, x)
        return x * torch.heaviside(x - theta, torch.tensor(0.0).cuda())

    @staticmethod
    def backward(cxt, grad_output):
        theta, x, = cxt.saved_tensors
        x_grad = torch.heaviside(x - theta, torch.tensor(0.0).cuda()) * grad_output
        theta_grad = -(theta / bandwidth) * rectangle((x - theta) / bandwidth) * grad_output
        return x_grad, theta_grad

class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(cxt, x, theta):
        cxt.save_for_backward(theta, x)
        return torch.heaviside(x - theta, torch.tensor(0.0).cuda())

    @staticmethod
    def backward(cxt, grad_output):
        theta, x, = cxt.saved_tensors
        x_grad = 0.0 * grad_output
        theta_grad = -(1.0/bandwidth) * rectangle((x - theta) / bandwidth) * grad_output
        return x_grad, theta_grad

jumprelu = JumpReluActivation.apply
step = StepFunction.apply

log_all = False

class JumpReluSAEModel(nn.Module):
    def __init__(self, d_model, n_features, sparsity_penalty):
        super().__init__()
        self.n_features = n_features
        self.w_enc = nn.Parameter(torch.randn(d_model, n_features)) # (D,F)
        self.w_dec = nn.Parameter(torch.randn(n_features, d_model)) # (F,D)
        self.b_enc = nn.Parameter(torch.randn(n_features))  # (F)
        self.b_dec = nn.Parameter(torch.randn(d_model)) # (D)
        self.log_theta = nn.Parameter(torch.randn(n_features)) # (F)
        self.sparsity_penalty = sparsity_penalty

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier uniform initialization for weights
        nn.init.xavier_uniform_(self.w_enc)
        nn.init.xavier_uniform_(self.w_dec)

        # Initialize biases to small values
        nn.init.uniform_(self.b_enc, -0.1, 0.1)
        nn.init.uniform_(self.b_dec, -0.1, 0.1)

        # Initialize log_theta to produce reasonable initial thresholds
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_enc)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.log_theta, math.log(bound), math.log(bound * 10))

    # returns (encoded->decoded x, feature vector) in order for the loss to work properly
    def forward(self, x):
        # encode
        x2 = torch.relu(x @ self.w_enc + self.b_enc)
        theta = torch.exp(self.log_theta)
        f = jumprelu(x2, theta)
        # decode
        xf = f @ self.w_dec + self.b_dec

        if log_all:
            print(f"x shape: {x.shape}, min: {x.min().item()}, max: {x.max().item()}")
            print(f"x2 shape: {x2.shape}, min: {x2.min().item()}, max: {x2.max().item()}")
            print(f"f shape: {f.shape}, min: {f.min().item()}, max: {f.max().item()}")
            print(f"xf shape: {xf.shape}, min: {xf.min().item()}, max: {xf.max().item()}")

        return xf, f

    # returns loss, reconstruction loss because having the reconstruction loss separate is convenient
    # okay now it also returns the n of active features
    def loss(self, x):
        xf, f = self(x)
        l_r = torch.square(x - xf).sum(-1)
        theta = torch.exp(self.log_theta)
        n_f = step(f, theta).sum(-1)
        l_s = self.sparsity_penalty * n_f    

        if log_all:
            print(f"l_r: {l_r.mean().item()}, l_s: {l_s.mean().item()}")
            print(f"n_f mean: {n_f.float().mean().item()}")
            print(f"theta min: {theta.min().item()}, max: {theta.max().item()}")

        return torch.mean(l_r + l_s), torch.mean(l_r), torch.mean(n_f)


def train_sae(model, sae, dataloader, optimizer, device, tokenizer, start_epoch=0, start_batch_idx=-1, total_loss=0, num_epochs=5):
    global log_all
    model.eval()
    sae.train()
    total_batches = len(dataloader)
    print(f"Total number of batches per epoch: {total_batches}")

    log = open("sae-log.csv", "a")

    for epoch in range(start_epoch, num_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        avg_n_f = 0
        avg_l_r = 0
        avg_l_t = 0

        for batch_idx, batch in progress_bar:
            if batch_idx <= start_batch_idx:
                continue

            input_ids = batch['input_ids'].to(device)
            with torch.no_grad():
                with model.trace(input_ids):
                    # This gets the output from the whole layer, i.e. after layernorm (and adding the residual)
                    # We should figure out whether that's actually the right way to do this
                    sae_input = model.transformer_layer.output.save()
            #log_all = (batch_idx) % 500 == 0

            optimizer.zero_grad()
            loss, rec_loss, n_f = sae.loss(sae_input)
            loss.backward()
            optimizer.step()

            cur_loss = loss.item()
            total_loss += cur_loss
            avg_n_f += n_f.item() / 500
            avg_l_r += rec_loss.item() / 500
            avg_l_t += loss.item() / 500

            # Update progress bar every 100 batches
            if (batch_idx) % 500 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'Batch': batch_idx, 'Avg Loss': f'{avg_loss:.4f}', 'Rec Loss': f'{avg_l_r:.4f}'})
                log.write(f'{epoch},{batch_idx},{avg_loss:.4f},{avg_l_t:.4f},{avg_l_r:.4f},{avg_n_f:.4f}\n')
                log.flush()
                save(sae, optimizer, epoch, batch_idx, total_loss, file='sae-checkpoint')
                avg_n_f = 0
                avg_l_r = 0
                avg_l_t = 0
                if batch_idx % 50000 == 0:
                    save(sae, optimizer, epoch, batch_idx, total_loss, file=f'sae-checkpoint-{epoch}-{batch_idx}')

                #sample_text = generate_text(model, tokenizer, device, "The financial market", max_length=50)
                #print(f"Sample generated text: {sample_text}")

        epoch_loss = total_loss / total_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Generate sample text after each epoch
        #sample_text = generate_text(model, tokenizer, device, "The financial market", max_length=50)
        #print(f"Sample generated text: {sample_text}")

def dictionary_learn(model, sae, dataloader, tokenizer, batch_size=32, n_activations=20):
    # We want to track the top `n_activations` activation contexts for each feature
    model.eval()
    sae.eval()
    feature_map = [[] for _ in range(sae.n_features)]
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Dictionary learning...")
        for batch_idx, batch in progress_bar:
            input_ids = batch['input_ids'].cuda()
            with model.trace(input_ids):
                sae_input = model.transformer_layer.output.save()
            xf, f = sae(sae_input)
            # I expect this to be pretty slow, but we'll see if it's actually a bottleneck or not
            # f is (B,L,F)
            f = f.cpu()
            for i in range(f.shape[-1]):
                for b in range(f.shape[0]):
                    # We save a whole sequence plus the corresponding feature vector to the feature map
                    vector = f[b,:,i]
                    activation = torch.max(vector).item()
                    saved = feature_map[i]
                    if len(saved) < n_activations:
                        text = tokenizer.decode(input_ids[b,:])
                        saved.append((activation, text, vector.contiguous()))
                        saved.sort(key=lambda x: x[0])
                    elif saved[0][0] < activation:
                        text = tokenizer.decode(input_ids[b,:])
                        saved[0] = (activation, text, vector.contiguous())
                        saved.sort(key=lambda x: x[0])
            if batch_idx % 500 == 0:
                torch.save(feature_map, f'feature_map_{batch_idx}')
    return feature_map

# Load and preprocess the dataset
def load_and_preprocess_data(tokenizer, max_length=128, batch_size=32):
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=16, load_from_cache_file=True)
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

# Saving and loading checkpoints
def save(model, optimizer, epoch, batch_idx, total_loss, file='sae-lm-checkpoint'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'batch_idx': batch_idx,
        'total_loss': total_loss,
    }, file)
def load(model, optimizer, file='sae-lm-checkpoint'):
    checkpoint = torch.load(file, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (checkpoint['epoch'], checkpoint['batch_idx'], checkpoint['total_loss'])

# Training function
# Updated training function with progress bar
def train(model, dataloader, optimizer, criterion, device, tokenizer, start_epoch=0, start_batch_idx=-1, total_loss=0, num_epochs=5):
    model.train()
    total_batches = len(dataloader)
    print(f"Total number of batches per epoch: {total_batches}")

    log = open("sae-lm-log.csv", "a")
    chunk_loss = 0

    for epoch in range(start_epoch, num_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in progress_bar:
            if batch_idx <= start_batch_idx:
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs[:, :-1, :].reshape(-1, outputs.size(-1)), input_ids[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            cur_loss = loss.item()
            total_loss += cur_loss
            chunk_loss += cur_loss

            # Update progress bar every 100 batches
            if (batch_idx + 1) % 500 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_loss_chunk = chunk_loss / 500
                progress_bar.set_postfix({'Batch': batch_idx, 'Avg Loss': f'{avg_loss:.4f}', 'Chunk Loss': f'{avg_loss_chunk:.4f}'})
                log.write(f'{epoch},{batch_idx},{avg_loss:.4f},{avg_loss_chunk:.4f}\n')
                log.flush()
                save(model, optimizer, epoch, batch_idx, total_loss)
                # save an additional backup checkpoint every 50000 batches (about every two hours)
                if (batch_idx + 1) % 50000 == 0:
                    save(model, optimizer, epoch, batch_idx, total_loss, file=f'sae-lm-checkpoint-{batch_idx+1}')
                chunk_loss = 0

                sample_text = generate_text(model, tokenizer, device, "The financial market", max_length=50)
                print(f"Sample generated text: {sample_text}")
                model.train()

        epoch_loss = total_loss / total_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Generate sample text after each epoch
        sample_text = generate_text(model, tokenizer, device, "The financial market", max_length=50)
        print(f"Sample generated text: {sample_text}")


# Text generation function
def generate_text(model, tokenizer, device, prompt, max_length=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, max_length, temperature)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Main function to set up and run the training
def main(mode="train", prompt="The economy is"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and add padding token
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model with updated vocabulary size
    model = SimpleTransformer(len(tokenizer)).to(device)
    print('Model has', sum(p.numel() for p in model.parameters()), 'parameters')

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    epoch = 0
    batch_idx = -1
    total_loss = 0

    # Load last checkpoint - use "restart" to start from scratch
    if mode != "restart":
        epoch, batch_idx, total_loss = load(model, optimizer)

    if mode == "train" or mode == "restart":
        # Prepare data
        dataloader = load_and_preprocess_data(tokenizer)

        # Train the model
        train(model, dataloader, optimizer, criterion, device, tokenizer, start_epoch=epoch, start_batch_idx=batch_idx, total_loss=total_loss)

        # Save the trained model
        torch.save(model.state_dict(), "sae-lm.pth")
        print("Training completed and model saved.")

    if mode == 'sae' or mode == 'dict':
        # Wrap it in nnsight
        nmodel = nnsight.models.NNsightModel.NNsight(model)
        print(nmodel)

        # Build SAE
        sae = JumpReluSAEModel(512, 512*4, 1e-3).cuda()
        optimizer = optim.Adam(sae.parameters(), lr=1e-3)

        epoch = 0
        batch_idx = -1
        total_loss = 0
        epoch, batch_idx, total_loss = load(sae, optimizer, file='sae-checkpoint')

        # Prepare data
        dataloader = load_and_preprocess_data(tokenizer)

        if mode == 'sae':
            train_sae(nmodel, sae, dataloader, optimizer, device, tokenizer, start_epoch=epoch, start_batch_idx=batch_idx, total_loss=total_loss)
        elif mode == 'dict':
            feature_map = dictionary_learn(nmodel, sae, dataloader, tokenizer)
            print(feature_map)
            torch.save(feature_map, 'feature_map_final')
    else:
        # Generate final sample text
        for i in range(5):
            final_sample = generate_text(model, tokenizer, device, prompt, max_length=100, temperature=1.0)
            print(f"Generated text: {final_sample}\n")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(mode=sys.argv[1], prompt=sys.argv[2])
    elif len(sys.argv) > 1:
        main(mode=sys.argv[1])
    else:
        main()
