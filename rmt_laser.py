import logging
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from lib.utils import gptq_data_utils
from tqdm import tqdm
import random
import numpy as np
import time

# Setup argument parsing
parser = argparse.ArgumentParser(description='Modify a transformer model.')
parser.add_argument('model_name', type=str, help='The name of the model to modify.')
parser.add_argument('profile_name', type=str, help='The name of your huggingface profile.')
parser.add_argument('hf_token', type=str, help='The token to access your huggingface account.')
args = parser.parse_args()

# Use the model_name argument to initialize your ModelModifier
model_name = args.model_name
model_name_only = model_name.split('/')[1]
profile = args.profile_name
token_value = args.hf_token

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("output.log", mode='a'), # Append mode
                        logging.StreamHandler() # Console output
                    ])

class ModelModifier:
    def __init__(self, model_name):
        # Determine the device as part of the class initialization
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.data_type = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.data_type = torch.float16
        else:
            self.device = torch.device("cpu")
            self.data_type = torch.float16  # Default to float16 for non-GPU devices for consistency

        logging.info(f"Using device: {self.device}")

        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.data_type, token=token_value).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token_value)
        self.original_weights = {}
        self.modified_layers = set()
        self.failed_attempts = set()


    def update_model_reduce_layer(self, layer_type, layer_number):
        layer_id = f"{layer_type}_{layer_number}"
        if layer_id in self.modified_layers:
            print(f"Layer {layer_id} has already been modified. Skipping.")
            logging.info(f"Layer {layer_id} has already been modified. Skipping.")
            return False

        for name, module in self.model.named_modules():
            if layer_type in name and str(layer_number) in name:
                print(f"Reconstructing layer: {name}")
                logging.info(f"Reconstructing layer: {name}")
                original_dtype = module.weight.dtype
                self.original_weights[name] = module.weight.detach().clone()
                is_cuda_available = torch.cuda.is_available()
                if is_cuda_available:
                # If CUDA is available, use double precision
                    weights = module.weight.double()
                else:
                    # If CUDA is not available (e.g., using MPS), use single precision (float32)
                    weights = module.weight.float()  # Convert to float32 for compatibility with MPS

                U, S, V = torch.linalg.svd(weights, full_matrices=False)

                # Estimate sigma using the full IQR method
                sigma_estimated_full_iqr = self.estimate_sigma_with_full_iqr(S)

                # Calculate Marchenko-Pastur threshold
                n, m = weights.shape
                mp_threshold_full_iqr = self.marchenko_pastur_threshold(sigma_estimated_full_iqr, n, m)

                # Retain only the singular values above the MP threshold
                S_reduced = torch.zeros_like(S)
                k = (S > mp_threshold_full_iqr).sum().item()
                S_reduced[:k] = S[:k]
                print(f"Reduced from {S.shape} to {k}")
                logging.info(f"Reduced from {S.shape} to {k}")

                # Reconstruct the matrix using the thresholded singular values
                reconstructed_weights = U @ torch.diag(S_reduced) @ V
                reconstructed_weights = reconstructed_weights.to(original_dtype)
                module.weight = torch.nn.Parameter(reconstructed_weights)
                self.modified_layers.add(layer_id)
                return True

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta))**2)
        return threshold

    ## Calculate an estimate of the standard deviation of the singular values based on Inter Quantile Range

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349 ## 0.6745 * sigma is the expected range between the quantiles (Q1 and Q3)
        return sigma_estimated


    def restore_model_original_layer(self, layer_type, layer_number):
        layer_id = f"{layer_type}_{layer_number}"
        for name, module in self.model.named_modules():
            if layer_type in name and layer_number in name:
                if name in self.original_weights:
                    module.weight = torch.nn.Parameter(self.original_weights[name])
                    print(f"Restored original weights for layer: {name}")
                    logging.info(f"Restored original weights for layer: {name}")
                    if layer_id in self.modified_layers:
                        self.modified_layers.remove(layer_id)
                else:
                    print(f"No original weights saved for layer: {name}")
                    logging.info(f"No original weights saved for layer: {name}")


#removed ", 'c4'" for faster tests
    def calculate_model_perplexity(self, datasets=['wikitext2', 'wikitext_de', 'ptb'], seqlen=384, use_cuda_graph=False, use_flash_attn=False):
      start_time = time.time()
      model = self.model
      acc_loss = 0.0
      total_samples = 0

      for dataset in datasets:
        dataset_start_time = time.time()
        input_tok = gptq_data_utils.get_test_tokens(dataset, seed=0, seqlen=seqlen, model=self.model_name)

        # Check the shape and adjust if necessary
        if input_tok.shape[1] != seqlen:
            if input_tok.shape[1] > seqlen:
                # Assuming the tensor might be too large and needs to be split into [nsamples, seqlen]
                try:
                    nsamples = input_tok.shape[1] // seqlen
                    input_tok = input_tok[0, :nsamples * seqlen].view(nsamples, seqlen)
                except Exception as e:
                    raise ValueError(f"Cannot reshape tensor from {input_tok.shape} to [{nsamples}, {seqlen}]: {str(e)}")
            else:
                raise ValueError(f"Expected sequence length {seqlen}, but got {input_tok.shape[1]} for dataset {dataset}")

        nsamples = input_tok.shape[0]
        total_samples += nsamples
        logging.info(f"Processing dataset {dataset}. Total samples: {nsamples}")

        loss_fct = torch.nn.CrossEntropyLoss().to(self.device)
        progress = tqdm(range(nsamples), desc=f"Processing {dataset}")

        for ii in progress:
            input = input_tok[ii, :].to(self.device).unsqueeze(0)
            output = model(input, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss / (ii + 1)}")

        dataset_elapsed = time.time() - dataset_start_time
        logging.info(f"Completed processing dataset {dataset} in {dataset_elapsed:.2f} seconds.")

      avg_loss = acc_loss / total_samples
      ppl = torch.exp(torch.tensor(avg_loss)).item()
      total_elapsed = time.time() - start_time
      logging.info(f"Completed perplexity calculation for all datasets in {total_elapsed:.2f} seconds. Total samples processed: {total_samples}. Perplexity: {ppl}")

      return ppl

  
    def calculate_model_perplexity_old(self, datasets=['wikitext2', 'wikitext_de', 'ptb'], seqlen=384, use_cuda_graph=False, use_flash_attn=False):
      start_time = time.time()

      model = self.model
      acc_loss = 0.0
      total_samples = 0

      for dataset in datasets:
        dataset_start_time = time.time()
        input_tok = gptq_data_utils.get_test_tokens(dataset, seed=0, seqlen=seqlen, model=self.model_name)
        nsamples = input_tok.numel() // seqlen
        input_tok = input_tok[0, :(seqlen * nsamples)].view(nsamples, seqlen)
        total_samples += nsamples

        logging.info(f"Processing dataset {dataset}. Total samples: {nsamples}")

        loss_fct = torch.nn.CrossEntropyLoss().to(self.device)
        progress = tqdm(range(nsamples), desc=f"Processing {dataset}")

        for ii in progress:
            input = input_tok[ii, :].to(self.device).view(1, -1)
            output = model(input, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss / (ii + 1)}")

        dataset_elapsed = time.time() - dataset_start_time
        logging.info(f"Completed processing dataset {dataset} in {dataset_elapsed:.2f} seconds.")

      avg_loss = acc_loss / total_samples
      ppl = torch.exp(torch.tensor(avg_loss)).item()

      total_elapsed = time.time() - start_time
      logging.info(f"Completed perplexity calculation for all datasets in {total_elapsed:.2f} seconds. Total samples processed: {total_samples}. Perplexity: {ppl}")

      return ppl

    ### Implement a Backward Search
    # Search for the optimal lower ranking approximations from the top layers downwards
    # Also, we try doing a greedy approach, in order to maximize the rank reduction.
    # We tune the compression rate based on Marchenko-Pastur Random Matrix Theory
    ######################################################################################

    def search_optimal_layer_modification(self, layer_types, layer_numbers, max_mod=5):
        start_time = time.time()  # Start time of the operation
        total_operations = len(layer_types) * len(layer_numbers)
        operations_completed = 0
        # Calculate initial perplexity with original model weights
        initial_perplexity = self.calculate_model_perplexity()
        print("="*50)
        print(f"The initial perplexity of the model is {initial_perplexity}")
        print("="*50)
        logging.info("="*50)
        logging.info(f"The initial perplexity of the model is {initial_perplexity}")
        logging.info("="*50)
        logging.info(f"Starting optimization over {total_operations} potential modifications.")


        min_loss = initial_perplexity
        optimal_params = (None, None)
        mods = 0

        for layer_number in layer_numbers:
            for layer_type in layer_types:
                operations_completed += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                estimated_total_time = (elapsed_time / operations_completed) * total_operations
                remaining_time = estimated_total_time - elapsed_time

                logging.info(f"Optimizing layer {layer_type} {layer_number}. Progress: {operations_completed}/{total_operations}. Estimated time remaining: {remaining_time:.2f} seconds.")

                if mods >= max_mod and max_mod != -1:
                    return optimal_params, min_loss
                attempt = (layer_type, layer_number)
                if attempt in self.failed_attempts:
                    continue  # Skip this attempt if it has failed before

                try_update = self.update_model_reduce_layer(layer_type, layer_number)

                if not try_update:
                    continue  # Skip this attempt if it has already been modified before

                try:
                    loss = self.calculate_model_perplexity()
                    if loss < min_loss:
                        min_loss = loss
                        optimal_params = (layer_type, layer_number)
                        mods = mods + 1
                        # Break out of the loop as soon as a better configuration is found
                        print("*"*50)
                        print(f"Improved perplexity found: {min_loss} for layer {layer_type} {layer_number}. Total modifications is {mods}")
                        print("*"*50)

                        logging.info("*"*50)
                        logging.info(f"Improved perplexity found: {min_loss} for layer {layer_type} {layer_number}. Total modifications is {mods}")
                        logging.info("*"*50)

                        layer_number_str = str(layer_number)
                        logging.info("temp saving: "+layer_number_str+" ...\n")
                        self.save_model("laser_model_"+layer_number_str)
                    else:
                        self.restore_model_original_layer(layer_type, layer_number)
                        self.failed_attempts.add(attempt)  # Record the failed attempt

                except NotImplementedError:
                    print("Perplexity calculation method is not implemented yet.")
                    logging.info("Perplexity calculation method is not implemented yet.")
                    return False, min_loss

        logging.info("Optimization complete. Final model evaluation in progress.")
        return optimal_params, min_loss

    def save_model(self, save_dir):

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

# Usage
modifier = ModelModifier(model_name)


# %%
## Sequence of tests

modifier.update_model_reduce_layer('mlp.down_proj', '25')
modifier.update_model_reduce_layer('mlp.down_proj', '25')
modifier.restore_model_original_layer('mlp.down_proj', '25')
modifier.update_model_reduce_layer('mlp.down_proj', '25')
modifier.restore_model_original_layer('mlp.down_proj', '25')

# %%
layers = list(range(31, -1, -1))
layers = [f".{l}." for l in layers]
print(layers)
logging.info("\nlayers:\n")
logging.info(layers)

# %%
## Search and modify all layers

loop_check, min_loss = modifier.search_optimal_layer_modification(layer_types=['mlp.down_proj', 'mlp.up_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'],
                                    layer_numbers=layers)

# %%
logging.info("saving...")
modifier.save_model("laser_model_final")

logging.info("uploading...")

from huggingface_hub import HfApi, create_repo

api = HfApi(token=token_value)
repo_name = profile + "/" + model_name_only + "-laser"

try:
    # Attempt to fetch the repository details.
    repo_info = api.repo_info(repo_name)
    print(f"Repository '{repo_name}' already exists.")
except:
    # If the repository does not exist, create it.
    create_repo(repo_name, token=token_value)
    print(f"Repository '{repo_name}' has been created.")

api = HfApi(token=token_value)
api.upload_folder(
    repo_id=repo_name,
    folder_path="laser_model_final"
)
